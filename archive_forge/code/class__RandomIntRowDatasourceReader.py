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
class _RandomIntRowDatasourceReader(Reader):

    def __init__(self, n: int, num_columns: int):
        self._n = n
        self._num_columns = num_columns

    def estimate_inmemory_data_size(self) -> Optional[int]:
        return self._n * self._num_columns * 8

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        _check_pyarrow_version()
        import pyarrow
        read_tasks: List[ReadTask] = []
        n = self._n
        num_columns = self._num_columns
        block_size = max(1, n // parallelism)

        def make_block(count: int, num_columns: int) -> Block:
            return pyarrow.Table.from_arrays(np.random.randint(np.iinfo(np.int64).max, size=(num_columns, count), dtype=np.int64), names=[f'c_{i}' for i in range(num_columns)])
        schema = pyarrow.Table.from_pydict({f'c_{i}': [0] for i in range(num_columns)}).schema
        i = 0
        while i < n:
            count = min(block_size, n - i)
            meta = BlockMetadata(num_rows=count, size_bytes=8 * count * num_columns, schema=schema, input_files=None, exec_stats=None)
            read_tasks.append(ReadTask(lambda count=count, num_columns=num_columns: [make_block(count, num_columns)], meta))
            i += block_size
        return read_tasks