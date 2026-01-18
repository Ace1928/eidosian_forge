import abc
import contextlib
import copy
import inspect
import os
import posixpath
import shutil
import textwrap
import time
import urllib
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple, Union
from unittest.mock import patch
import fsspec
import pyarrow as pa
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config, utils
from .arrow_dataset import Dataset
from .arrow_reader import (
from .arrow_writer import ArrowWriter, BeamWriter, ParquetWriter, SchemaInferenceError
from .data_files import DataFilesDict, DataFilesPatternsDict, sanitize_patterns
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager, DownloadMode
from .download.mock_download_manager import MockDownloadManager
from .download.streaming_download_manager import StreamingDownloadManager, xjoin, xopen
from .exceptions import DatasetGenerationCastError, DatasetGenerationError, FileFormatError, ManualDownloadError
from .features import Features
from .filesystems import (
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict, PostProcessedInfo
from .iterable_dataset import ArrowExamplesIterable, ExamplesIterable, IterableDataset
from .keyhash import DuplicatedKeysError
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH, camelcase_to_snakecase
from .splits import Split, SplitDict, SplitGenerator, SplitInfo
from .streaming import extend_dataset_builder_for_streaming
from .table import CastError
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils._filelock import FileLock
from .utils.file_utils import cached_path, is_remote_url
from .utils.info_utils import VerificationMode, get_size_checksum_dict, verify_checksums, verify_splits
from .utils.py_utils import (
from .utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs
from .utils.track import tracked_list
def _check_legacy_cache(self) -> Optional[str]:
    """Check for the old cache directory template {cache_dir}/{namespace}___{builder_name} from 2.13"""
    if self.__module__.startswith('datasets.') and (not is_remote_url(self._cache_dir_root)) and (self.config.name == 'default'):
        from .packaged_modules import _PACKAGED_DATASETS_MODULES
        namespace = self.repo_id.split('/')[0] if self.repo_id and self.repo_id.count('/') > 0 else None
        config_name = self.repo_id.replace('/', '--') if self.repo_id is not None else self.dataset_name
        config_id = config_name + self.config_id[len(self.config.name):]
        hash = _PACKAGED_DATASETS_MODULES.get(self.name, 'missing')[1]
        legacy_relative_data_dir = posixpath.join(self.dataset_name if namespace is None else f'{namespace}___{self.dataset_name}', config_id, '0.0.0', hash)
        legacy_cache_dir = posixpath.join(self._cache_dir_root, legacy_relative_data_dir)
        if os.path.isdir(legacy_cache_dir):
            return legacy_relative_data_dir