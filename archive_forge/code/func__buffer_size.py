from typing import Optional
from ray.data._internal.arrow_block import ArrowBlockAccessor
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
def _buffer_size(self) -> int:
    """Return shuffle buffer size."""
    buffer_size = self._builder.num_rows()
    buffer_size += self._materialized_buffer_size()
    return buffer_size