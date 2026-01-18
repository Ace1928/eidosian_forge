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
def _prepare_split(self, split_generator, pipeline, file_format='arrow', max_shard_size: Optional[Union[str, int]]=None):
    import apache_beam as beam
    if max_shard_size is not None:
        raise NotImplementedError('max_shard_size is not supported for Beam datasets.Please set it to None to use the default Apache Beam sharding and get the best performance.')
    split_name = split_generator.split_info.name
    fname = f'{self.dataset_name}-{split_name}.{file_format}'
    fpath = posixpath.join(self._output_dir, fname)
    beam_writer = BeamWriter(features=self.info.features, path=fpath, namespace=split_name, cache_dir=self._output_dir)
    self._beam_writers[split_name] = beam_writer
    encode_example = self.info.features.encode_example

    @beam.ptransform_fn
    def _build_pcollection(pipeline):
        """PTransformation which build a single split."""
        pcoll_examples = self._build_pcollection(pipeline, **split_generator.gen_kwargs)
        pcoll_examples |= 'Encode' >> beam.Map(lambda key_ex: (key_ex[0], encode_example(key_ex[1])))
        return beam_writer.write_from_pcollection(pcoll_examples)
    _ = pipeline | split_name >> _build_pcollection()