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
def _download_and_prepare(self, dl_manager, verification_mode, **prepare_splits_kwargs):
    import apache_beam as beam
    import datasets.utils.beam_utils as beam_utils
    beam_runner = self._beam_runner
    beam_options = self._beam_options
    if not beam_runner and (not beam_options):
        usage_example = f"load_dataset('{self.name}', '{self.config.name}', beam_runner='DirectRunner')"
        raise MissingBeamOptions(f'Trying to generate a dataset using Apache Beam, yet no Beam Runner or PipelineOptions() has been provided in `load_dataset` or in the builder arguments. For big datasets it has to run on large-scale data processing tools like Dataflow, Spark, etc. More information about Apache Beam runners at https://beam.apache.org/documentation/runners/capability-matrix/\nIf you really want to run it locally because you feel like the Dataset is small enough, you can use the local beam runner called `DirectRunner` (you may run out of memory). \nExample of usage: \n\t`{usage_example}`')
    if self._writer_batch_size is not None:
        logger.warning('`writer_batch_size` is not supported for beam pipelines yet. Using the default chunk size for writing.')
    pipeline_options = {'pipeline_type_check': False}
    if 'num_proc' in prepare_splits_kwargs:
        num_workers = prepare_splits_kwargs.pop('num_proc')
        pipeline_options['direct_num_workers'] = num_workers
        pipeline_options['num_workers'] = num_workers
        pipeline_options['direct_running_mode'] = 'multi_processing'
        raise NotImplementedError('Using a DirectRunner with `num_proc` for multiprocessing it not supported yet.')
    beam_options = beam_options or beam.options.pipeline_options.PipelineOptions.from_dictionary(pipeline_options)
    pipeline = beam_utils.BeamPipeline(runner=beam_runner, options=beam_options)
    super()._download_and_prepare(dl_manager, verification_mode=VerificationMode.NO_CHECKS, pipeline=pipeline, **prepare_splits_kwargs)
    pipeline_results = pipeline.run()
    pipeline_results.wait_until_finish()
    metrics = pipeline_results.metrics()
    split_dict = self.info.splits
    for split_name, beam_writer in self._beam_writers.items():
        m_filter = beam.metrics.MetricsFilter().with_namespace(namespace=split_name)
        num_examples, num_bytes = beam_writer.finalize(metrics.query(m_filter))
        split_info = split_dict[split_name]
        split_info.num_examples = num_examples
        split_info.num_bytes = num_bytes
        if hasattr(beam_writer, '_shard_lengths') and len(beam_writer._shard_lengths) > 1:
            split_info.shard_lengths = beam_writer._shard_lengths
        else:
            file_format = prepare_splits_kwargs.get('file_format', 'arrow')
            src_fname = f'{self.dataset_name}-{split_name}-00000-of-00001.{file_format}'
            dst_fname = f'{self.dataset_name}-{split_name}.{file_format}'
            src_fpath = posixpath.join(self._output_dir, src_fname)
            dst_fpath = posixpath.join(self._output_dir, dst_fname)
            self._rename(src_fpath, dst_fpath)