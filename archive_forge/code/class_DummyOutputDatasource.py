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
@DeveloperAPI
class DummyOutputDatasource(Datasource):
    """An example implementation of a writable datasource for testing.

    Examples:
        >>> import ray
        >>> from ray.data.datasource import DummyOutputDatasource
        >>> output = DummyOutputDatasource() # doctest: +SKIP
        >>> ray.data.range(10).write_datasource(output) # doctest: +SKIP
        >>> assert output.num_ok == 1 # doctest: +SKIP
    """

    def __init__(self):
        ctx = DataContext.get_current()

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
        self.data_sink = DataSink.remote()
        self.num_ok = 0
        self.num_failed = 0
        self.enabled = True

    def write(self, blocks: Iterable[Block], ctx: TaskContext, **write_args) -> WriteResult:
        tasks = []
        if not self.enabled:
            raise ValueError('disabled')
        for b in blocks:
            tasks.append(self.data_sink.write.remote(b))
        ray.get(tasks)
        return 'ok'

    def on_write_complete(self, write_results: List[WriteResult]) -> None:
        assert all((w == 'ok' for w in write_results)), write_results
        self.num_ok += 1

    def on_write_failed(self, write_results: List[ObjectRef[WriteResult]], error: Exception) -> None:
        self.num_failed += 1