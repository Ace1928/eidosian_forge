import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
class BlocksToBatchesMapTransformFn(MapTransformFn):
    """A MapTransformFn that converts input blocks to batches."""

    def __init__(self, batch_size: Optional[int]=None, batch_format: str='default', zero_copy_batch: bool=False):
        self._batch_size = batch_size
        self._batch_format = batch_format
        self._ensure_copy = not zero_copy_batch and batch_size is not None
        super().__init__(MapTransformFnDataType.Block, MapTransformFnDataType.Batch)

    def __call__(self, blocks: Iterable[Block], _: TaskContext) -> Iterable[DataBatch]:
        """Converts input blocks to batches."""
        block_iter = iter(blocks)
        first = next(block_iter, None)
        if first is None:
            return []
        blocks = itertools.chain([first], block_iter)
        empty_block = BlockAccessor.for_block(first).builder().build()
        first = None
        formatted_batch_iter = batch_blocks(blocks=blocks, stats=None, batch_size=self._batch_size, batch_format=self._batch_format, ensure_copy=self._ensure_copy)
        first = next(formatted_batch_iter, None)
        if first is None:
            return [empty_block]
        else:
            return itertools.chain([first], formatted_batch_iter)

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def batch_format(self) -> str:
        return self._batch_format

    @property
    def zero_copy_batch(self) -> bool:
        return not self._ensure_copy

    def __repr__(self) -> str:
        return f'BlocksToBatchesMapTransformFn(batch_size={self._batch_size}, batch_format={self._batch_format}, zero_copy_batch={self.zero_copy_batch})'