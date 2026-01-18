from typing import Optional
from ray.data._internal.arrow_block import ArrowBlockAccessor
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
class Batcher(BatcherInterface):
    """Chunks blocks into batches."""

    def __init__(self, batch_size: Optional[int], ensure_copy: bool=False):
        """
        Construct a batcher that yields batches of batch_sizes rows.

        Args:
            batch_size: The size of batches to yield.
            ensure_copy: Whether batches are always copied from the underlying base
                blocks (not zero-copy views).
        """
        self._batch_size = batch_size
        self._buffer = []
        self._buffer_size = 0
        self._done_adding = False
        self._ensure_copy = ensure_copy

    def add(self, block: Block):
        """Add a block to the block buffer.

        Note empty block is not added to buffer.

        Args:
            block: Block to add to the block buffer.
        """
        if BlockAccessor.for_block(block).num_rows() > 0:
            self._buffer.append(block)
            self._buffer_size += BlockAccessor.for_block(block).num_rows()

    def done_adding(self) -> bool:
        """Indicate to the batcher that no more blocks will be added to the batcher."""
        self._done_adding = True

    def has_batch(self) -> bool:
        """Whether this Batcher has any full batches."""
        return self.has_any() and (self._batch_size is None or self._buffer_size >= self._batch_size)

    def has_any(self) -> bool:
        """Whether this Batcher has any data."""
        return self._buffer_size > 0

    def next_batch(self) -> Block:
        """Get the next batch from the block buffer.

        Returns:
            A batch represented as a Block.
        """
        assert self.has_batch() or (self._done_adding and self.has_any())
        needs_copy = self._ensure_copy
        if self._batch_size is None:
            assert len(self._buffer) == 1
            block = self._buffer[0]
            if needs_copy:
                block = BlockAccessor.for_block(block)
                block = block.slice(0, block.num_rows(), copy=True)
            self._buffer = []
            self._buffer_size = 0
            return block
        output = DelegatingBlockBuilder()
        leftover = []
        needed = self._batch_size
        for block in self._buffer:
            accessor = BlockAccessor.for_block(block)
            if needed <= 0:
                leftover.append(block)
            elif accessor.num_rows() <= needed:
                output.add_block(accessor.slice(0, accessor.num_rows(), copy=False))
                needed -= accessor.num_rows()
            else:
                if isinstance(accessor, ArrowBlockAccessor) and block.num_columns > 0 and (block.column(0).num_chunks >= MIN_NUM_CHUNKS_TO_TRIGGER_COMBINE_CHUNKS):
                    accessor = BlockAccessor.for_block(transform_pyarrow.combine_chunks(block))
                output.add_block(accessor.slice(0, needed, copy=False))
                leftover.append(accessor.slice(needed, accessor.num_rows(), copy=False))
                needed = 0
        self._buffer = leftover
        self._buffer_size -= self._batch_size
        needs_copy = needs_copy and (not output.will_build_yield_copy())
        batch = output.build()
        if needs_copy:
            batch = BlockAccessor.for_block(batch)
            batch = batch.slice(0, batch.num_rows(), copy=True)
        return batch