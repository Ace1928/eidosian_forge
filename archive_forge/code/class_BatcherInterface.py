from typing import Optional
from ray.data._internal.arrow_block import ArrowBlockAccessor
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
class BatcherInterface:

    def add(self, block: Block):
        """Add a block to the block buffer.

        Args:
            block: Block to add to the block buffer.
        """
        raise NotImplementedError()

    def done_adding(self) -> bool:
        """Indicate to the batcher that no more blocks will be added to the buffer."""
        raise NotImplementedError()

    def has_batch(self) -> bool:
        """Whether this Batcher has any full batches."""
        raise NotImplementedError()

    def has_any(self) -> bool:
        """Whether this Batcher has any data."""
        raise NotImplementedError()

    def next_batch(self) -> Block:
        """Get the next batch from the block buffer.

        Returns:
            A batch represented as a Block.
        """
        raise NotImplementedError()