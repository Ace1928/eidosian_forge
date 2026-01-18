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
def _check_log_dir_exists(self) -> None:
    if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
        rank_zero_warn(f'Experiment logs directory {self.log_dir} exists and is not empty. Previous log files in this directory will be deleted when the new ones are saved!')
        if self._fs.isfile(self.metrics_file_path):
            self._fs.rm_file(self.metrics_file_path)