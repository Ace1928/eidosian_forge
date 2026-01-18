import itertools
import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
import ibis
from ibis import BaseBackend
from triad import assert_or_throw
from fugue import StructuredRawSQL
from fugue.bag import Bag, LocalBag
from fugue.collections.partition import (
from fugue.dataframe import DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.utils import get_join_schemas
from fugue.execution.execution_engine import ExecutionEngine, MapEngine, SQLEngine
from ._compat import IbisTable
from ._utils import to_ibis_schema
from .dataframe import IbisDataFrame
class IbisMapEngine(MapEngine):
    """IbisExecutionEngine's MapEngine, it is a wrapper of the map engine
    of :meth:`~.IbisExecutionEngine.non_ibis_engine`

    :param execution_engine: the execution engine this map engine will run on
    """

    @property
    def is_distributed(self) -> bool:
        return self._ibis_engine.non_ibis_engine.map_engine.is_distributed

    def __init__(self, execution_engine: ExecutionEngine) -> None:
        super().__init__(execution_engine)
        self._ibis_engine: IbisExecutionEngine = execution_engine

    @property
    def execution_engine_constraint(self) -> Type[ExecutionEngine]:
        return IbisExecutionEngine

    def map_dataframe(self, df: DataFrame, map_func: Callable[[PartitionCursor, LocalDataFrame], LocalDataFrame], output_schema: Any, partition_spec: PartitionSpec, on_init: Optional[Callable[[int, DataFrame], Any]]=None, map_func_format_hint: Optional[str]=None) -> DataFrame:
        _df = self._ibis_engine._to_non_ibis_dataframe(df)
        return self._ibis_engine.non_ibis_engine.map_engine.map_dataframe(_df, map_func=map_func, output_schema=output_schema, partition_spec=partition_spec, on_init=on_init, map_func_format_hint=map_func_format_hint)

    def map_bag(self, bag: Bag, map_func: Callable[[BagPartitionCursor, LocalBag], LocalBag], partition_spec: PartitionSpec, on_init: Optional[Callable[[int, Bag], Any]]=None) -> Bag:
        return self._ibis_engine.non_ibis_engine.map_engine.map_bag(bag, map_func, partition_spec, on_init)