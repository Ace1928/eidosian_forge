from typing import Optional
from ray.data._internal.arrow_block import ArrowBlockAccessor
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
def _materialized_buffer_size(self) -> int:
    """Return materialized (compacted portion of) shuffle buffer size."""
    if self._shuffle_buffer is None:
        return 0
    return max(0, BlockAccessor.for_block(self._shuffle_buffer).num_rows() - self._batch_head)