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
def _build_single_dataset(self, split: Union[str, ReadInstruction, Split], run_post_process: bool, verification_mode: VerificationMode, in_memory: bool=False):
    """as_dataset for a single split."""
    if not isinstance(split, ReadInstruction):
        split = str(split)
        if split == 'all':
            split = '+'.join(self.info.splits.keys())
        split = Split(split)
    ds = self._as_dataset(split=split, in_memory=in_memory)
    if run_post_process:
        for resource_file_name in self._post_processing_resources(split).values():
            if os.sep in resource_file_name:
                raise ValueError(f"Resources shouldn't be in a sub-directory: {resource_file_name}")
        resources_paths = {resource_name: os.path.join(self._output_dir, resource_file_name) for resource_name, resource_file_name in self._post_processing_resources(split).items()}
        post_processed = self._post_process(ds, resources_paths)
        if post_processed is not None:
            ds = post_processed
            recorded_checksums = {}
            record_checksums = False
            for resource_name, resource_path in resources_paths.items():
                size_checksum = get_size_checksum_dict(resource_path)
                recorded_checksums[resource_name] = size_checksum
            if verification_mode == VerificationMode.ALL_CHECKS and record_checksums:
                if self.info.post_processed is None or self.info.post_processed.resources_checksums is None:
                    expected_checksums = None
                else:
                    expected_checksums = self.info.post_processed.resources_checksums.get(split)
                verify_checksums(expected_checksums, recorded_checksums, 'post processing resources')
            if self.info.post_processed is None:
                self.info.post_processed = PostProcessedInfo()
            if self.info.post_processed.resources_checksums is None:
                self.info.post_processed.resources_checksums = {}
            self.info.post_processed.resources_checksums[str(split)] = recorded_checksums
            self.info.post_processing_size = sum((checksums_dict['num_bytes'] for split_checksums_dicts in self.info.post_processed.resources_checksums.values() for checksums_dict in split_checksums_dicts.values()))
            if self.info.dataset_size is not None and self.info.download_size is not None:
                self.info.size_in_bytes = self.info.dataset_size + self.info.download_size + self.info.post_processing_size
            self._save_info()
            ds._info.post_processed = self.info.post_processed
            ds._info.post_processing_size = self.info.post_processing_size
            ds._info.size_in_bytes = self.info.size_in_bytes
            if self.info.post_processed.features is not None:
                if self.info.post_processed.features.type != ds.features.type:
                    raise ValueError(f"Post-processed features info don't match the dataset:\nGot\n{self.info.post_processed.features}\nbut expected something like\n{ds.features}")
                else:
                    ds.info.features = self.info.post_processed.features
    return ds