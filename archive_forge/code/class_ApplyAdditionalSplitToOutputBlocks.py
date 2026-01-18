import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
class ApplyAdditionalSplitToOutputBlocks(MapTransformFn):
    """Do additional splits on output blocks."""

    def __init__(self, additional_split_factor: int):
        """
        Args:
          additional_output_splits: The number of additional splits, must be
          greater than 1.
        """
        assert additional_split_factor > 1
        self._additional_split_factor = additional_split_factor
        super().__init__(MapTransformFnDataType.Block, MapTransformFnDataType.Block)

    def __call__(self, blocks: Iterable[Block], ctx: TaskContext) -> Iterable[Block]:
        for block in blocks:
            block = BlockAccessor.for_block(block)
            offset = 0
            split_sizes = _splitrange(block.num_rows(), self._additional_split_factor)
            for size in split_sizes:
                yield block.slice(offset, offset + size, copy=True)
                offset += size