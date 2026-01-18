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
class CachedMetricModuleFactory(_MetricModuleFactory):
    """
    Get the module of a metric that has been loaded once already and cached.
    The script that is loaded from the cache is the most recent one with a matching name.

    <Deprecated version="2.5.0">

    Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate

    </Deprecated>
    """

    @deprecated('Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate')
    def __init__(self, name: str, dynamic_modules_path: Optional[str]=None):
        self.name = name
        self.dynamic_modules_path = dynamic_modules_path
        assert self.name.count('/') == 0

    def get_module(self) -> MetricModule:
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        importable_directory_path = os.path.join(dynamic_modules_path, 'metrics', self.name)
        hashes = [h for h in os.listdir(importable_directory_path) if len(h) == 64] if os.path.isdir(importable_directory_path) else None
        if not hashes:
            raise FileNotFoundError(f'Metric {self.name} is not cached in {dynamic_modules_path}')

        def _get_modification_time(module_hash):
            return (Path(importable_directory_path) / module_hash / (self.name + '.py')).stat().st_mtime
        hash = sorted(hashes, key=_get_modification_time)[-1]
        logger.warning(f"Using the latest cached version of the module from {os.path.join(importable_directory_path, hash)} (last modified on {time.ctime(_get_modification_time(hash))}) since it couldn't be found locally at {self.name}, or remotely on the Hugging Face Hub.")
        module_path = '.'.join([os.path.basename(dynamic_modules_path), 'metrics', self.name, hash, self.name])
        importlib.invalidate_caches()
        return MetricModule(module_path, hash)