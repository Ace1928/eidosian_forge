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
class CachedDatasetModuleFactory(_DatasetModuleFactory):
    """
    Get the module of a dataset that has been loaded once already and cached.
    The script that is loaded from the cache is the most recent one with a matching name.
    """

    def __init__(self, name: str, cache_dir: Optional[str]=None, dynamic_modules_path: Optional[str]=None):
        self.name = name
        self.cache_dir = cache_dir
        self.dynamic_modules_path = dynamic_modules_path
        assert self.name.count('/') <= 1

    def get_module(self) -> DatasetModule:
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        importable_directory_path = os.path.join(dynamic_modules_path, 'datasets', self.name.replace('/', '--'))
        hashes = [h for h in os.listdir(importable_directory_path) if len(h) == 64] if os.path.isdir(importable_directory_path) else None
        if hashes:

            def _get_modification_time(module_hash):
                return (Path(importable_directory_path) / module_hash / (self.name.split('/')[-1] + '.py')).stat().st_mtime
            hash = sorted(hashes, key=_get_modification_time)[-1]
            warning_msg = f"Using the latest cached version of the module from {os.path.join(importable_directory_path, hash)} (last modified on {time.ctime(_get_modification_time(hash))}) since it couldn't be found locally at {self.name}"
            if not config.HF_DATASETS_OFFLINE:
                warning_msg += ', or remotely on the Hugging Face Hub.'
            logger.warning(warning_msg)
            module_path = '.'.join([os.path.basename(dynamic_modules_path), 'datasets', self.name.replace('/', '--'), hash, self.name.split('/')[-1]])
            importlib.invalidate_caches()
            builder_kwargs = {'repo_id': self.name}
            return DatasetModule(module_path, hash, builder_kwargs)
        cache_dir = os.path.expanduser(str(self.cache_dir or config.HF_DATASETS_CACHE))
        cached_datasets_directory_path_root = os.path.join(cache_dir, self.name.replace('/', '___'))
        cached_directory_paths = [cached_directory_path for cached_directory_path in glob.glob(os.path.join(cached_datasets_directory_path_root, '*', '*', '*')) if os.path.isdir(cached_directory_path)]
        if cached_directory_paths:
            builder_kwargs = {'repo_id': self.name, 'dataset_name': self.name.split('/')[-1]}
            warning_msg = f"Using the latest cached version of the dataset since {self.name} couldn't be found on the Hugging Face Hub"
            if config.HF_DATASETS_OFFLINE:
                warning_msg += ' (offline mode is enabled).'
            logger.warning(warning_msg)
            return DatasetModule('datasets.packaged_modules.cache.cache', 'auto', {**builder_kwargs, 'version': 'auto'})
        raise FileNotFoundError(f'Dataset {self.name} is not cached in {self.cache_dir}')