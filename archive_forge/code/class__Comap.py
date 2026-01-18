import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
from uuid import uuid4
from triad import ParamDict, Schema, SerializableRLock, assert_or_throw, to_uuid
from triad.collections.function_wrapper import AnnotatedParam
from triad.exceptions import InvalidOperationError
from triad.utils.convert import to_size
from fugue.bag import Bag, LocalBag
from fugue.collections.partition import (
from fugue.collections.sql import StructuredRawSQL, TempTableName
from fugue.collections.yielded import PhysicalYielded, Yielded
from fugue.column import (
from fugue.constants import _FUGUE_GLOBAL_CONF, FUGUE_SQL_DEFAULT_DIALECT
from fugue.dataframe import AnyDataFrame, DataFrame, DataFrames, fugue_annotated_param
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.dataframe.dataframe import LocalDataFrame
from fugue.dataframe.utils import deserialize_df, serialize_df
from fugue.exceptions import FugueWorkflowRuntimeError
class _Comap:

    def __init__(self, df: DataFrame, key_schema: Schema, func: Callable, output_schema: Schema, on_init: Optional[Callable[[int, DataFrames], Any]]):
        self.schemas = df.metadata['schemas']
        self.key_schema = key_schema
        self.output_schema = output_schema
        self.dfs_count = len(self.schemas)
        self.named = df.metadata.get_or_throw('serialized_has_name', bool)
        self.func = func
        self.how = df.metadata.get_or_throw('serialized_join_how', str)
        self._on_init = on_init

    def on_init(self, partition_no, df: DataFrame) -> None:
        if self._on_init is None:
            return
        empty_dfs = _generate_comap_empty_dfs(self.schemas, self.named)
        self._on_init(partition_no, empty_dfs)

    def run(self, cursor: PartitionCursor, df: LocalDataFrame) -> LocalDataFrame:
        data = df.as_dicts()
        if self.how == 'inner':
            if len(data) < self.dfs_count:
                return ArrayDataFrame([], self.output_schema)
        elif self.how == 'left_outer':
            if data[0][_FUGUE_SERIALIZED_BLOB_NO_COL] > 0:
                return ArrayDataFrame([], self.output_schema)
        elif self.how == 'right_outer':
            if data[-1][_FUGUE_SERIALIZED_BLOB_NO_COL] != self.dfs_count - 1:
                return ArrayDataFrame([], self.output_schema)
        dfs = self._get_dfs(data)
        _c = PartitionSpec(by=self.key_schema.names).get_cursor(dfs[0].schema, cursor.physical_partition_no)
        _c.set(lambda: dfs[0].peek_array(), cursor.partition_no, cursor.slice_no)
        return self.func(_c, dfs)

    def _get_dfs(self, rows: List[Dict[str, Any]]) -> DataFrames:
        tdfs: Dict[Any, DataFrame] = {}
        for row in rows:
            df = deserialize_df(row[_FUGUE_SERIALIZED_BLOB_COL])
            if df is not None:
                if self.named:
                    tdfs[row[_FUGUE_SERIALIZED_BLOB_NAME_COL]] = df
                else:
                    tdfs[row[_FUGUE_SERIALIZED_BLOB_NO_COL]] = df
        dfs: Dict[Any, DataFrame] = {}
        for k, schema in self.schemas.items():
            if k in tdfs:
                dfs[k] = tdfs[k]
            else:
                dfs[k] = ArrayDataFrame([], schema)
        return DataFrames(dfs) if self.named else DataFrames(list(dfs.values()))