from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
import pyspark.sql as ps
from pyspark.sql.functions import col
from triad import SerializableRLock
from triad.collections.schema import SchemaError
from triad.utils.assertion import assert_or_throw
from fugue.dataframe import (
from fugue.dataframe.utils import pa_table_as_array, pa_table_as_dicts
from fugue.exceptions import FugueDataFrameOperationError
from fugue.plugins import (
from ._utils.convert import (
from ._utils.misc import is_spark_connect, is_spark_dataframe
class SparkDataFrame(DataFrame):
    """DataFrame that wraps Spark DataFrame. Please also read
    |DataFrameTutorial| to understand this Fugue concept

    :param df: :class:`spark:pyspark.sql.DataFrame`
    :param schema: |SchemaLikeObject| or :class:`spark:pyspark.sql.types.StructType`,
      defaults to None.

    .. note::

        * You should use :meth:`fugue_spark.execution_engine.SparkExecutionEngine.to_df`
          instead of construction it by yourself.
        * If ``schema`` is set, then there will be type cast on the Spark DataFrame if
          the schema is different.
    """

    def __init__(self, df: Any=None, schema: Any=None):
        self._lock = SerializableRLock()
        if is_spark_dataframe(df):
            if schema is not None:
                schema = to_schema(schema).assert_not_empty()
                has_cast, expr = to_cast_expression(df, schema, True)
                if has_cast:
                    df = df.selectExpr(*expr)
            else:
                schema = to_schema(df).assert_not_empty()
            self._native = df
            super().__init__(schema)
        else:
            assert_or_throw(schema is not None, SchemaError('schema is None'))
            schema = to_schema(schema).assert_not_empty()
            raise ValueError(f'{df} is incompatible with SparkDataFrame')

    @property
    def alias(self) -> str:
        return '_' + str(id(self.native))

    @property
    def native(self) -> ps.DataFrame:
        """The wrapped Spark DataFrame

        :rtype: :class:`spark:pyspark.sql.DataFrame`
        """
        return self._native

    def native_as_df(self) -> ps.DataFrame:
        return self._native

    @property
    def is_local(self) -> bool:
        return False

    @property
    def is_bounded(self) -> bool:
        return True

    def as_local_bounded(self) -> LocalBoundedDataFrame:
        res = ArrowDataFrame(self.as_arrow())
        if self.has_metadata:
            res.reset_metadata(self.metadata)
        return res

    @property
    def num_partitions(self) -> int:
        return _spark_num_partitions(self.native)

    @property
    def empty(self) -> bool:
        return self._first is None

    def peek_array(self) -> List[Any]:
        self.assert_not_empty()
        return self._first

    def count(self) -> int:
        with self._lock:
            if '_df_count' not in self.__dict__:
                self._df_count = self.native.count()
            return self._df_count

    def _drop_cols(self, cols: List[str]) -> DataFrame:
        cols = (self.schema - cols).names
        return self._select_cols(cols)

    def _select_cols(self, cols: List[Any]) -> DataFrame:
        schema = self.schema.extract(cols)
        return SparkDataFrame(self.native[schema.names])

    def as_pandas(self) -> pd.DataFrame:
        return _spark_df_as_pandas(self.native)

    def as_arrow(self, type_safe: bool=False) -> pa.Table:
        return _spark_df_as_arrow(self.native)

    def rename(self, columns: Dict[str, str]) -> DataFrame:
        try:
            self.schema.rename(columns)
        except Exception as e:
            raise FugueDataFrameOperationError from e
        return SparkDataFrame(_rename_spark_dataframe(self.native, columns))

    def alter_columns(self, columns: Any) -> DataFrame:
        new_schema = self.schema.alter(columns)
        if new_schema == self.schema:
            return self
        return SparkDataFrame(self.native, new_schema)

    def as_array(self, columns: Optional[List[str]]=None, type_safe: bool=False) -> List[Any]:
        return _spark_as_array(self.native, columns=columns, type_safe=type_safe)

    def as_array_iterable(self, columns: Optional[List[str]]=None, type_safe: bool=False) -> Iterable[Any]:
        yield from _spark_as_array_iterable(self.native, columns=columns, type_safe=type_safe)

    def as_dicts(self, columns: Optional[List[str]]=None) -> List[Dict[str, Any]]:
        return _spark_as_dicts(self.native, columns=columns)

    def as_dict_iterable(self, columns: Optional[List[str]]=None) -> Iterable[Dict[str, Any]]:
        yield from _spark_as_dict_iterable(self.native, columns=columns)

    def head(self, n: int, columns: Optional[List[str]]=None) -> LocalBoundedDataFrame:
        sdf = self._select_columns(columns)
        return SparkDataFrame(sdf.native.limit(n), sdf.schema).as_local()

    @property
    def _first(self) -> Optional[List[Any]]:
        with self._lock:
            if '_first_row' not in self.__dict__:
                self._first_row = self.native.first()
                if self._first_row is not None:
                    self._first_row = list(self._first_row)
            return self._first_row

    def _select_columns(self, columns: Optional[List[str]]) -> 'SparkDataFrame':
        if columns is None:
            return self
        return SparkDataFrame(self.native.select(*columns))