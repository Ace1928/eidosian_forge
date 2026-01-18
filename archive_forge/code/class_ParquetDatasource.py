import logging
from typing import (
import numpy as np
import ray
import ray.cloudpickle as cloudpickle
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.util import _check_pyarrow_version, _is_local_scheme
from ray.data.block import Block
from ray.data.context import DataContext
from ray.data.datasource import Datasource
from ray.data.datasource._default_metadata_providers import (
from ray.data.datasource.datasource import ReadTask
from ray.data.datasource.file_meta_provider import (
from ray.data.datasource.partitioning import PathPartitionFilter
from ray.data.datasource.path_util import (
from ray.util.annotations import PublicAPI
@PublicAPI
class ParquetDatasource(Datasource):
    """Parquet datasource, for reading and writing Parquet files.

    The primary difference from ParquetBaseDatasource is that this uses
    PyArrow's `ParquetDataset` abstraction for dataset reads, and thus offers
    automatic Arrow dataset schema inference and row count collection at the
    cost of some potential performance and/or compatibility penalties.
    """

    def __init__(self, paths: Union[str, List[str]], *, columns: Optional[List[str]]=None, dataset_kwargs: Optional[Dict[str, Any]]=None, to_batch_kwargs: Optional[Dict[str, Any]]=None, _block_udf: Optional[Callable[[Block], Block]]=None, filesystem: Optional['pyarrow.fs.FileSystem']=None, schema: Optional[Union[type, 'pyarrow.lib.Schema']]=None, meta_provider: ParquetMetadataProvider=DefaultParquetMetadataProvider(), partition_filter: PathPartitionFilter=None, shuffle: Union[Literal['files'], None]=None, include_paths: bool=False, file_extensions: Optional[List[str]]=None):
        _check_pyarrow_version()
        import pyarrow as pa
        import pyarrow.parquet as pq
        self._supports_distributed_reads = not _is_local_scheme(paths)
        if not self._supports_distributed_reads and ray.util.client.ray.is_connected():
            raise ValueError("Because you're using Ray Client, read tasks scheduled on the Ray cluster can't access your local files. To fix this issue, store files in cloud storage or a distributed filesystem like NFS.")
        self._local_scheduling = None
        if not self._supports_distributed_reads:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
            self._local_scheduling = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False)
        paths, filesystem = _resolve_paths_and_filesystem(paths, filesystem)
        if partition_filter is not None or file_extensions is not None:
            default_meta_provider = get_generic_metadata_provider(file_extensions=None)
            expanded_paths, _ = map(list, zip(*default_meta_provider.expand_paths(paths, filesystem)))
            paths = list(expanded_paths)
            if partition_filter is not None:
                paths = partition_filter(paths)
            if file_extensions is not None:
                paths = [path for path in paths if _has_file_extension(path, file_extensions)]
            filtered_paths = set(expanded_paths) - set(paths)
            if filtered_paths:
                logger.info(f'Filtered out {len(filtered_paths)} paths')
        elif len(paths) == 1:
            paths = paths[0]
        if dataset_kwargs is None:
            dataset_kwargs = {}
        try:
            pq_ds = pq.ParquetDataset(paths, **dataset_kwargs, filesystem=filesystem, use_legacy_dataset=False)
        except OSError as e:
            _handle_read_os_error(e, paths)
        if schema is None:
            schema = pq_ds.schema
        if columns:
            schema = pa.schema([schema.field(column) for column in columns], schema.metadata)
        if _block_udf is not None:
            dummy_table = schema.empty_table()
            try:
                inferred_schema = _block_udf(dummy_table).schema
                inferred_schema = inferred_schema.with_metadata(schema.metadata)
            except Exception:
                logger.debug('Failed to infer schema of dataset by passing dummy table through UDF due to the following exception:', exc_info=True)
                inferred_schema = schema
        else:
            inferred_schema = schema
        try:
            prefetch_remote_args = {}
            if self._local_scheduling:
                prefetch_remote_args['scheduling_strategy'] = self._local_scheduling
            self._metadata = meta_provider.prefetch_file_metadata(pq_ds.fragments, **prefetch_remote_args) or []
        except OSError as e:
            _handle_read_os_error(e, paths)
        if to_batch_kwargs is None:
            to_batch_kwargs = {}
        self._pq_fragments = [_SerializedFragment(p) for p in pq_ds.fragments]
        self._pq_paths = [p.path for p in pq_ds.fragments]
        self._meta_provider = meta_provider
        self._inferred_schema = inferred_schema
        self._block_udf = _block_udf
        self._to_batches_kwargs = to_batch_kwargs
        self._columns = columns
        self._schema = schema
        self._encoding_ratio = self._estimate_files_encoding_ratio()
        self._file_metadata_shuffler = None
        self._include_paths = include_paths
        if shuffle == 'files':
            self._file_metadata_shuffler = np.random.default_rng()

    def estimate_inmemory_data_size(self) -> Optional[int]:
        total_size = 0
        for file_metadata in self._metadata:
            for row_group_idx in range(file_metadata.num_row_groups):
                row_group_metadata = file_metadata.row_group(row_group_idx)
                total_size += row_group_metadata.total_byte_size
        return total_size * self._encoding_ratio

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        pq_metadata = self._metadata
        if len(pq_metadata) < len(self._pq_fragments):
            pq_metadata += [None] * (len(self._pq_fragments) - len(pq_metadata))
        if self._file_metadata_shuffler is not None:
            files_metadata = list(zip(self._pq_fragments, self._pq_paths, pq_metadata))
            shuffled_files_metadata = [files_metadata[i] for i in self._file_metadata_shuffler.permutation(len(files_metadata))]
            pq_fragments, pq_paths, pq_metadata = list(map(list, zip(*shuffled_files_metadata)))
        else:
            pq_fragments, pq_paths, pq_metadata = (self._pq_fragments, self._pq_paths, pq_metadata)
        read_tasks = []
        for fragments, paths, metadata in zip(np.array_split(pq_fragments, parallelism), np.array_split(pq_paths, parallelism), np.array_split(pq_metadata, parallelism)):
            if len(fragments) <= 0:
                continue
            meta = self._meta_provider(paths, self._inferred_schema, num_fragments=len(fragments), prefetched_metadata=metadata)
            if self._to_batches_kwargs.get('filter') is not None:
                meta.num_rows = None
            if meta.size_bytes is not None:
                meta.size_bytes = int(meta.size_bytes * self._encoding_ratio)
            if meta.num_rows and meta.size_bytes:
                row_size = meta.size_bytes / meta.num_rows
                max_parquet_reader_row_batch_size_bytes = DataContext.get_current().target_max_block_size // 10
                default_read_batch_size_rows = max(1, min(PARQUET_READER_ROW_BATCH_SIZE, max_parquet_reader_row_batch_size_bytes // row_size))
            else:
                default_read_batch_size_rows = PARQUET_READER_ROW_BATCH_SIZE
            block_udf, to_batches_kwargs, columns, schema, include_paths = (self._block_udf, self._to_batches_kwargs, self._columns, self._schema, self._include_paths)
            read_tasks.append(ReadTask(lambda f=fragments: _read_fragments(block_udf, to_batches_kwargs, default_read_batch_size_rows, columns, schema, f, include_paths), meta))
        return read_tasks

    def _estimate_files_encoding_ratio(self) -> float:
        """Return an estimate of the Parquet files encoding ratio.

        To avoid OOMs, it is safer to return an over-estimate than an underestimate.
        """
        if not DataContext.get_current().decoding_size_estimation:
            return PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT
        num_files = len(self._pq_fragments)
        num_samples = int(num_files * PARQUET_ENCODING_RATIO_ESTIMATE_SAMPLING_RATIO)
        min_num_samples = min(PARQUET_ENCODING_RATIO_ESTIMATE_MIN_NUM_SAMPLES, num_files)
        max_num_samples = min(PARQUET_ENCODING_RATIO_ESTIMATE_MAX_NUM_SAMPLES, num_files)
        num_samples = max(min(num_samples, max_num_samples), min_num_samples)
        file_samples = [self._pq_fragments[idx] for idx in np.linspace(0, num_files - 1, num_samples).astype(int).tolist()]
        sample_fragment = cached_remote_fn(_sample_fragment)
        futures = []
        scheduling = self._local_scheduling or 'SPREAD'
        for sample in file_samples:
            futures.append(sample_fragment.options(scheduling_strategy=scheduling, retry_exceptions=[OSError]).remote(self._to_batches_kwargs, self._columns, self._schema, sample))
        sample_bar = ProgressBar('Parquet Files Sample', len(futures))
        sample_ratios = sample_bar.fetch_until_complete(futures)
        sample_bar.close()
        ratio = np.mean(sample_ratios)
        logger.debug(f'Estimated Parquet encoding ratio from sampling is {ratio}.')
        return max(ratio, PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND)

    def get_name(self):
        """Return a human-readable name for this datasource.
        This will be used as the names of the read tasks.
        Note: overrides the base `ParquetBaseDatasource` method.
        """
        return 'Parquet'

    @property
    def supports_distributed_reads(self) -> bool:
        return self._supports_distributed_reads