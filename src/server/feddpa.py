from pathlib import Path
import os
from src.utils.tools import (
    Logger
)
from rich.console import Console

from omegaconf import DictConfig

from src.client.feddpa import FedDpaClient
from src.server.fedavg import FedAvgServer


class FedDpaServer(FedAvgServer):
    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedDpa",
        unique_model: bool = False,
        use_fedavg_client_cls: bool = True,
        return_diff: bool = False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        # Initialize trainer with FedDpaClient
        self.init_trainer(FedDpaClient)

        # Get the parent directory
        parent_dir = self.output_dir.parent
        start = str(self.dataset).find('.') + 1  # Find the position of the first '.'
        end = str(self.dataset).find('object')  # Find the position of ' object'
        dataset_name = str(self.dataset)[start:end].split('.')[-1].lower()  # Extract the last part
        self.output_dir = parent_dir / Path(dataset_name + '_' + str(self.args.feddpa.fisher_threshold))

        if not os.path.isdir(self.output_dir) and (
            self.args.common.save_log
            or self.args.common.save_learning_curve_plot
            or self.args.common.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)

        stdout = Console(log_path=False, log_time=False, soft_wrap=True, tab_size=4)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.common.save_log,
            logfile_path=self.output_dir / "main.log",
        )