import filecmp
import glob
import importlib
import inspect
import json
import os
import posixpath
import shutil
import signal
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import fsspec
import requests
import yaml
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HfFileSystem
from . import config
from .arrow_dataset import Dataset
from .builder import BuilderConfig, DatasetBuilder
from .data_files import (
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager, xbasename, xglob, xjoin
from .exceptions import DataFilesNotFoundError, DatasetNotFoundError
from .features import Features
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict
from .iterable_dataset import IterableDataset
from .metric import Metric
from .naming import camelcase_to_snakecase, snakecase_to_camelcase
from .packaged_modules import (
from .splits import Split
from .utils import _datasets_server
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.file_utils import (
from .utils.hub import hf_hub_url
from .utils.info_utils import VerificationMode, is_small_dataset
from .utils.logging import get_logger
from .utils.metadata import MetadataConfigs
from .utils.py_utils import get_imports
from .utils.version import Version
def _create_importable_file(local_path: str, local_imports: List[Tuple[str, str]], additional_files: List[Tuple[str, str]], dynamic_modules_path: str, module_namespace: str, subdirectory_name: str, name: str, download_mode: DownloadMode) -> None:
    importable_directory_path = os.path.join(dynamic_modules_path, module_namespace, name.replace('/', '--'))
    Path(importable_directory_path).mkdir(parents=True, exist_ok=True)
    (Path(importable_directory_path).parent / '__init__.py').touch(exist_ok=True)
    importable_local_file = _copy_script_and_other_resources_in_importable_dir(name=name.split('/')[-1], importable_directory_path=importable_directory_path, subdirectory_name=subdirectory_name, original_local_path=local_path, local_imports=local_imports, additional_files=additional_files, download_mode=download_mode)
    logger.debug(f'Created importable dataset file at {importable_local_file}')