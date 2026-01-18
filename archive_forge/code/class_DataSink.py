import warnings
from typing import Any, Callable, Iterable, List, Optional
import numpy as np
import ray
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
@ray.remote(scheduling_strategy=ctx.scheduling_strategy)
class DataSink:

    def __init__(self):
        self.rows_written = 0
        self.enabled = True

    def write(self, block: Block) -> str:
        block = BlockAccessor.for_block(block)
        self.rows_written += block.num_rows()
        return 'ok'

    def get_rows_written(self):
        return self.rows_written