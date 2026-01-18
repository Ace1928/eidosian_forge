import errno
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from . import config
from .features import Features, Image, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .info import DatasetInfo
from .keyhash import DuplicatedKeysError, KeyHasher
from .table import array_cast, cast_array_to_feature, embed_table_storage, table_cast
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import hash_url_to_filename
from .utils.py_utils import asdict, first_non_null_value
class BeamWriter:
    """
    Shuffles and writes Examples to Arrow files.
    The Arrow files are converted from Parquet files that are the output of Apache Beam pipelines.
    """

    def __init__(self, features: Optional[Features]=None, schema: Optional[pa.Schema]=None, path: Optional[str]=None, namespace: Optional[str]=None, cache_dir: Optional[str]=None):
        if features is None and schema is None:
            raise ValueError('At least one of features and schema must be provided.')
        if path is None:
            raise ValueError('Path must be provided.')
        if features is not None:
            self._features: Features = features
            self._schema: pa.Schema = features.arrow_schema
        else:
            self._schema: pa.Schema = schema
            self._features: Features = Features.from_arrow_schema(schema)
        self._path = path
        self._parquet_path = os.path.splitext(path)[0]
        self._namespace = namespace or 'default'
        self._num_examples = None
        self._cache_dir = cache_dir or config.HF_DATASETS_CACHE

    def write_from_pcollection(self, pcoll_examples):
        """Add the final steps of the beam pipeline: write to parquet files."""
        import apache_beam as beam

        def inc_num_examples(example):
            beam.metrics.Metrics.counter(self._namespace, 'num_examples').inc()
        _ = pcoll_examples | 'Count N. Examples' >> beam.Map(inc_num_examples)
        return pcoll_examples | 'Get values' >> beam.Values() | 'Save to parquet' >> beam.io.parquetio.WriteToParquet(self._parquet_path, self._schema, shard_name_template='-SSSSS-of-NNNNN.parquet')

    def finalize(self, metrics_query_result: dict):
        """
        Run after the pipeline has finished.
        It converts the resulting parquet files to arrow and it completes the info from the pipeline metrics.

        Args:
            metrics_query_result: `dict` obtained from pipeline_results.metrics().query(m_filter). Make sure
                that the filter keeps only the metrics for the considered split, under the namespace `split_name`.
        """
        fs, _, [parquet_path] = fsspec.get_fs_token_paths(self._parquet_path)
        parquet_path = str(Path(parquet_path)) if not is_remote_filesystem(fs) else fs.unstrip_protocol(parquet_path)
        shards = fs.glob(parquet_path + '*.parquet')
        num_bytes = sum(fs.sizes(shards))
        shard_lengths = get_parquet_lengths(shards)
        if self._path.endswith('.arrow'):
            logger.info(f'Converting parquet files {self._parquet_path} to arrow {self._path}')
            try:
                num_bytes = 0
                for shard in hf_tqdm(shards, unit='shards'):
                    with fs.open(shard, 'rb') as source:
                        with fs.open(shard.replace('.parquet', '.arrow'), 'wb') as destination:
                            shard_num_bytes, _ = parquet_to_arrow(source, destination)
                            num_bytes += shard_num_bytes
            except OSError as e:
                if e.errno != errno.EPIPE:
                    raise
                logger.warning('Broken Pipe during stream conversion from parquet to arrow. Using local convert instead')
                local_convert_dir = os.path.join(self._cache_dir, 'beam_convert')
                os.makedirs(local_convert_dir, exist_ok=True)
                num_bytes = 0
                for shard in hf_tqdm(shards, unit='shards'):
                    local_parquet_path = os.path.join(local_convert_dir, hash_url_to_filename(shard) + '.parquet')
                    fs.download(shard, local_parquet_path)
                    local_arrow_path = local_parquet_path.replace('.parquet', '.arrow')
                    shard_num_bytes, _ = parquet_to_arrow(local_parquet_path, local_arrow_path)
                    num_bytes += shard_num_bytes
                    remote_arrow_path = shard.replace('.parquet', '.arrow')
                    fs.upload(local_arrow_path, remote_arrow_path)
        counters_dict = {metric.key.metric.name: metric.result for metric in metrics_query_result['counters']}
        self._num_examples = counters_dict['num_examples']
        self._num_bytes = num_bytes
        self._shard_lengths = shard_lengths
        return (self._num_examples, self._num_bytes)