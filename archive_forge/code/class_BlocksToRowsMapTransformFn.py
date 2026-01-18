import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
class BlocksToRowsMapTransformFn(MapTransformFn):
    """A MapTransformFn that converts input blocks to rows."""

    def __init__(self):
        super().__init__(MapTransformFnDataType.Block, MapTransformFnDataType.Row)

    def __call__(self, blocks: Iterable[Block], _: TaskContext) -> Iterable[Row]:
        for block in blocks:
            block = BlockAccessor.for_block(block)
            for row in block.iter_rows(public_row_format=True):
                yield row

    @classmethod
    def instance(cls) -> 'BlocksToRowsMapTransformFn':
        """Returns the singleton instance."""
        if getattr(cls, '_instance', None) is None:
            cls._instance = cls()
        return cls._instance

    def __repr__(self) -> str:
        return 'BlocksToRowsMapTransformFn()'