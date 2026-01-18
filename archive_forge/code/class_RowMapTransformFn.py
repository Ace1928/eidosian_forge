import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
class RowMapTransformFn(MapTransformFn):
    """A rows-to-rows MapTransformFn."""

    def __init__(self, row_fn: MapTransformCallable[Row, Row]):
        self._row_fn = row_fn
        super().__init__(MapTransformFnDataType.Row, MapTransformFnDataType.Row)

    def __call__(self, input: Iterable[Row], ctx: TaskContext) -> Iterable[Row]:
        yield from self._row_fn(input, ctx)

    def __repr__(self) -> str:
        return f'RowMapTransformFn({self._row_fn})'