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
class PackagedDatasetModuleFactory(_DatasetModuleFactory):
    """Get the dataset builder module from the ones that are packaged with the library: csv, json, etc."""

    def __init__(self, name: str, data_dir: Optional[str]=None, data_files: Optional[Union[str, List, Dict]]=None, download_config: Optional[DownloadConfig]=None, download_mode: Optional[Union[DownloadMode, str]]=None):
        self.name = name
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_config = download_config
        self.download_mode = download_mode
        increase_load_count(name, resource_type='dataset')

    def get_module(self) -> DatasetModule:
        base_path = Path(self.data_dir or '').expanduser().resolve().as_posix()
        patterns = sanitize_patterns(self.data_files) if self.data_files is not None else get_data_patterns(base_path)
        data_files = DataFilesDict.from_patterns(patterns, download_config=self.download_config, base_path=base_path)
        supports_metadata = self.name in _MODULE_SUPPORTS_METADATA
        if self.data_files is None and supports_metadata and (patterns != DEFAULT_PATTERNS_ALL):
            try:
                metadata_patterns = get_metadata_patterns(base_path, download_config=self.download_config)
            except FileNotFoundError:
                metadata_patterns = None
            if metadata_patterns is not None:
                metadata_data_files_list = DataFilesList.from_patterns(metadata_patterns, download_config=self.download_config, base_path=base_path)
                if metadata_data_files_list:
                    data_files = DataFilesDict({split: data_files_list + metadata_data_files_list for split, data_files_list in data_files.items()})
        module_path, hash = _PACKAGED_DATASETS_MODULES[self.name]
        builder_kwargs = {'data_files': data_files, 'dataset_name': self.name}
        return DatasetModule(module_path, hash, builder_kwargs)