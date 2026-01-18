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
class _LegacyDatasourceReader(Reader):

    def __init__(self, datasource: Datasource, **read_args):
        self._datasource = datasource
        self._read_args = read_args

    def estimate_inmemory_data_size(self) -> Optional[int]:
        return None

    def get_read_tasks(self, parallelism: int) -> List['ReadTask']:
        return self._datasource.prepare_read(parallelism, **self._read_args)