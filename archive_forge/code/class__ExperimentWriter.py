import csv
import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Set, Union
from torch import Tensor
from typing_extensions import override
from lightning_fabric.loggers.logger import Logger, rank_zero_experiment
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.logger import _add_prefix
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.types import _PATH
class _ExperimentWriter:
    """Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs

    """
    NAME_METRICS_FILE = 'metrics.csv'

    def __init__(self, log_dir: str) -> None:
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []
        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)
        self._check_log_dir_exists()
        self._fs.makedirs(self.log_dir, exist_ok=True)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int]=None) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value
        if step is None:
            step = len(self.metrics)
        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics['step'] = step
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return
        new_keys = self._record_new_keys()
        file_exists = self._fs.isfile(self.metrics_file_path)
        if new_keys and file_exists:
            self._rewrite_with_new_header(self.metrics_keys)
        with self._fs.open(self.metrics_file_path, mode='a' if file_exists else 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_keys)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.metrics)
        self.metrics = []

    def _record_new_keys(self) -> Set[str]:
        """Records new keys that have not been logged before."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        self.metrics_keys.sort()
        return new_keys

    def _rewrite_with_new_header(self, fieldnames: List[str]) -> None:
        with self._fs.open(self.metrics_file_path, 'r', newline='') as file:
            metrics = list(csv.DictReader(file))
        with self._fs.open(self.metrics_file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

    def _check_log_dir_exists(self) -> None:
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero_warn(f'Experiment logs directory {self.log_dir} exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!')
            if self._fs.isfile(self.metrics_file_path):
                self._fs.rm_file(self.metrics_file_path)