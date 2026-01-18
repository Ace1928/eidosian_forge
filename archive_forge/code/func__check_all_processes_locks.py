import os
import types
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
from filelock import BaseFileLock, Timeout
from . import config
from .arrow_dataset import Dataset
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager
from .features import Features
from .info import DatasetInfo, MetricInfo
from .naming import camelcase_to_snakecase
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger
from .utils.py_utils import copyfunc, temp_seed
def _check_all_processes_locks(self):
    expected_lock_file_names = [os.path.join(self.data_dir, f'{self.experiment_id}-{self.num_process}-{process_id}.arrow.lock') for process_id in range(self.num_process)]
    for expected_lock_file_name in expected_lock_file_names:
        nofilelock = FileFreeLock(expected_lock_file_name)
        try:
            nofilelock.acquire(timeout=self.timeout)
        except Timeout:
            raise ValueError(f"Expected to find locked file {expected_lock_file_name} from process {self.process_id} but it doesn't exist.") from None
        else:
            nofilelock.release()