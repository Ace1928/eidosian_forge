import itertools
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union
from ray.data._internal.block_batching.block_batching import batch_blocks
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.output_buffer import BlockOutputBuffer
from ray.data.block import Block, BlockAccessor, DataBatch
class MapTransformFnDataType(Enum):
    """An enum that represents the input/output data type of a MapTransformFn."""
    Block = 0
    Row = 1
    Batch = 2