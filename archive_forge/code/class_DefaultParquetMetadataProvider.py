import itertools
import logging
import os
import pathlib
import re
from typing import (
import numpy as np
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockMetadata
from ray.data.datasource.partitioning import Partitioning
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
class DefaultParquetMetadataProvider(ParquetMetadataProvider):
    """The default file metadata provider for ParquetDatasource.

    Aggregates total block bytes and number of rows using the Parquet file metadata
    associated with a list of Arrow Parquet dataset file fragments.
    """

    def _get_block_metadata(self, paths: List[str], schema: Optional[Union[type, 'pyarrow.lib.Schema']], *, num_fragments: int, prefetched_metadata: Optional[List['pyarrow.parquet.FileMetaData']]) -> BlockMetadata:
        if prefetched_metadata is not None and len(prefetched_metadata) == num_fragments and all((m is not None for m in prefetched_metadata)):
            block_metadata = BlockMetadata(num_rows=sum((m.num_rows for m in prefetched_metadata)), size_bytes=sum((sum((m.row_group(i).total_byte_size for i in range(m.num_row_groups))) for m in prefetched_metadata)), schema=schema, input_files=paths, exec_stats=None)
        else:
            block_metadata = BlockMetadata(num_rows=None, size_bytes=None, schema=schema, input_files=paths, exec_stats=None)
        return block_metadata

    def prefetch_file_metadata(self, fragments: List['pyarrow.dataset.ParquetFileFragment'], **ray_remote_args) -> Optional[List['pyarrow.parquet.FileMetaData']]:
        from ray.data.datasource.parquet_datasource import FRAGMENTS_PER_META_FETCH, PARALLELIZE_META_FETCH_THRESHOLD, _fetch_metadata, _fetch_metadata_serialization_wrapper, _SerializedFragment
        if len(fragments) > PARALLELIZE_META_FETCH_THRESHOLD:
            fragments = [_SerializedFragment(fragment) for fragment in fragments]
            return list(_fetch_metadata_parallel(fragments, _fetch_metadata_serialization_wrapper, FRAGMENTS_PER_META_FETCH, **ray_remote_args))
        else:
            return _fetch_metadata(fragments)