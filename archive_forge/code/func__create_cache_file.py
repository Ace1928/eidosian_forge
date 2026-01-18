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
def _create_cache_file(self, timeout=1) -> Tuple[str, FileLock]:
    """Create a new cache file. If the default cache file is used, we generated a new hash."""
    file_path = os.path.join(self.data_dir, f'{self.experiment_id}-{self.num_process}-{self.process_id}.arrow')
    filelock = None
    for i in range(self.max_concurrent_cache_files):
        filelock = FileLock(file_path + '.lock')
        try:
            filelock.acquire(timeout=timeout)
        except Timeout:
            if self.num_process != 1:
                raise ValueError(f'Error in _create_cache_file: another metric instance is already using the local cache file at {file_path}. Please specify an experiment_id (currently: {self.experiment_id}) to avoid collision between distributed metric instances.') from None
            if i == self.max_concurrent_cache_files - 1:
                raise ValueError(f'Cannot acquire lock, too many metric instance are operating concurrently on this file system.You should set a larger value of max_concurrent_cache_files when creating the metric (current value is {self.max_concurrent_cache_files}).') from None
            file_uuid = str(uuid.uuid4())
            file_path = os.path.join(self.data_dir, f'{self.experiment_id}-{file_uuid}-{self.num_process}-{self.process_id}.arrow')
        else:
            break
    return (file_path, filelock)