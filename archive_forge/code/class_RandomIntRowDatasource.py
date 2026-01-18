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
class RandomIntRowDatasource(Datasource):
    """An example datasource that generates rows with random int64 columns.

    Examples:
        >>> import ray
        >>> from ray.data.datasource import RandomIntRowDatasource
        >>> source = RandomIntRowDatasource() # doctest: +SKIP
        >>> ray.data.read_datasource( # doctest: +SKIP
        ...     source, n=10, num_columns=2).take()
        {'c_0': 1717767200176864416, 'c_1': 999657309586757214}
        {'c_0': 4983608804013926748, 'c_1': 1160140066899844087}
    """

    def get_name(self) -> str:
        """Return a human-readable name for this datasource.
        This will be used as the names of the read tasks.
        Note: overrides the base `Datasource` method.
        """
        return 'RandomInt'

    def create_reader(self, n: int, num_columns: int) -> List[ReadTask]:
        return _RandomIntRowDatasourceReader(n, num_columns)