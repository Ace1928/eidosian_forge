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
def _sample_fragment(to_batches_kwargs, columns, schema, file_fragment: _SerializedFragment) -> float:
    fragment = _deserialize_fragments_with_retry([file_fragment])[0]
    fragment = fragment.subset(row_group_ids=[0])
    batch_size = max(min(fragment.metadata.num_rows, PARQUET_ENCODING_RATIO_ESTIMATE_NUM_ROWS), 1)
    to_batches_kwargs.pop('batch_size', None)
    batches = fragment.to_batches(columns=columns, schema=schema, batch_size=batch_size, **to_batches_kwargs)
    try:
        batch = next(batches)
    except StopIteration:
        ratio = PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND
    else:
        if batch.num_rows > 0:
            in_memory_size = batch.nbytes / batch.num_rows
            metadata = fragment.metadata
            total_size = 0
            for idx in range(metadata.num_row_groups):
                total_size += metadata.row_group(idx).total_byte_size
            file_size = total_size / metadata.num_rows
            ratio = in_memory_size / file_size
        else:
            ratio = PARQUET_ENCODING_RATIO_ESTIMATE_LOWER_BOUND
    logger.debug(f'Estimated Parquet encoding ratio is {ratio} for fragment {fragment} with batch size {batch_size}.')
    return ratio