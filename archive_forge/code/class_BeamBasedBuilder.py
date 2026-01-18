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
class BeamBasedBuilder(DatasetBuilder):
    """Beam-based Builder."""

    def __init__(self, *args, beam_runner=None, beam_options=None, **kwargs):
        self._beam_runner = beam_runner
        self._beam_options = beam_options
        self._beam_writers = {}
        super().__init__(*args, **kwargs)

    def _make_split_generators_kwargs(self, prepare_split_kwargs):
        split_generators_kwargs = {}
        split_generators_arg_names = inspect.signature(self._split_generators).parameters.keys()
        if 'pipeline' in split_generators_arg_names:
            split_generators_kwargs['pipeline'] = prepare_split_kwargs['pipeline']
        return split_generators_kwargs

    @abc.abstractmethod
    def _build_pcollection(self, pipeline, **kwargs):
        """Build the beam pipeline examples for each `SplitGenerator`.

        This function extracts examples from the raw data with parallel transforms
        in a Beam pipeline. It is called once for each `SplitGenerator` defined in
        `_split_generators`. The examples from the PCollection will be
        encoded and written to disk.

        <Tip warning={true}>
        Warning: When running in a distributed setup, make sure that the data
        which will be read (download_dir, manual_dir,...) and written (cache_dir)
        can be accessed by the workers jobs. The data should be located in a
        shared filesystem, like GCS.
        </Tip>

        Args:
            pipeline ([`utils.beam_utils.BeamPipeline`]):
                Apache Beam pipeline.
            **kwargs (additional keyword arguments):
                Arguments forwarded from the SplitGenerator.gen_kwargs.

        Returns:
            `beam.PCollection`: Apache Beam PCollection containing the
                example to send to `self.info.features.encode_example(...)`.

        Example:

        ```
        def _build_pcollection(pipeline, extracted_dir=None):
            return (
                    pipeline
                    | beam.Create(gfile.io.listdir(extracted_dir))
                    | beam.Map(_process_file)
            )
        ```
        """
        raise NotImplementedError()

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

    def _save_info(self):
        download_config = self.dl_manager.download_config if self.dl_manager else DownloadConfig(token=self.token, storage_options=self._fs.storage_options)
        with xopen(f'{self._output_dir}/{config.DATASET_INFO_FILENAME}', 'wb', download_config=download_config) as f:
            self.info._dump_info(f)
        if self.info.license:
            with xopen(f'{self._output_dir}/{config.LICENSE_FILENAME}', 'wb', download_config=download_config) as f:
                self.info._dump_license(f)

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

    def as_streaming_dataset(self, split: Optional[str]=None) -> Union[Dict[str, IterableDataset], IterableDataset]:
        self._request_info_from_hf_gcs()
        datasets = {split.name: IterableDataset(self._get_examples_iterable_for_split(split), info=self.info, split=split.name) for split in self.info.splits.values()}
        if split:
            try:
                datasets = datasets[split]
            except KeyError:
                raise ValueError(f'Bad split: {split}. Available splits: {list(datasets)}')
        if isinstance(datasets, dict):
            datasets = IterableDatasetDict(datasets)
        return datasets

    def _get_examples_iterable_for_split(self, split: SplitInfo) -> ExamplesIterable:
        return ExamplesIterable(self._generate_examples_from_hf_gcs, {'split': split})

    def _generate_examples_from_hf_gcs(self, split: SplitInfo):
        if split.shard_lengths:
            num_shards = len(split.shard_lengths)
            remote_prepared_urls = [f'{self._remote_cache_dir_from_hf_gcs}/{self.name}-{split.name}-{shard_id:05d}-of-{num_shards:05d}.arrow' for shard_id in range(num_shards)]
        else:
            remote_prepared_urls = [f'{self._remote_cache_dir_from_hf_gcs}/{self.name}-{split.name}.arrow']
        key = 0
        download_config = self.dl_manager.download_config if self.dl_manager else DownloadConfig(token=self.token, storage_options=self._fs.storage_options)
        for remote_prepared_url in remote_prepared_urls:
            with xopen(remote_prepared_url, 'rb', download_config=download_config) as f:
                with pa.ipc.open_stream(f) as reader:
                    for record_batch in reader:
                        for record in record_batch.to_pylist():
                            yield (key, record)
                            key += 1

    def _request_info_from_hf_gcs(self):
        from .download.streaming_download_manager import xopen
        remote_dataset_info = f'{self._remote_cache_dir_from_hf_gcs}/{config.DATASET_INFO_FILENAME}'
        try:
            download_config = download_config = self.dl_manager.download_config if self.dl_manager else DownloadConfig(token=self.token, storage_options=self._fs.storage_options)
            with xopen(remote_dataset_info, download_config=download_config) as f:
                import json
                _info = json.load(f)
        except FileNotFoundError as err:
            raise DatasetNotOnHfGcsError(err) from None
        self.info.update(DatasetInfo.from_dict(_info))

    @property
    def _remote_cache_dir_from_hf_gcs(self):
        relative_data_dir = self._relative_data_dir(with_hash=False)
        return HF_GCP_BASE_URL + '/' + Path(relative_data_dir).as_posix()