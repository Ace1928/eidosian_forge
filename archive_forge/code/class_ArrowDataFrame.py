from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import cast_pa_table, pa_table_to_pandas
from fugue.dataset.api import (
from fugue.exceptions import FugueDataFrameOperationError
from .api import (
from .dataframe import DataFrame, LocalBoundedDataFrame, _input_schema
from .utils import (
class ArrowDataFrame(LocalBoundedDataFrame):
    """DataFrame that wraps :func:`pyarrow.Table <pa:pyarrow.table>`. Please also read
    |DataFrameTutorial| to understand this Fugue concept

    :param df: 2-dimensional array, iterable of arrays,
      :func:`pyarrow.Table <pa:pyarrow.table>` or pandas DataFrame
    :param schema: |SchemaLikeObject|

    .. admonition:: Examples

        >>> ArrowDataFrame([[0,'a'],[1,'b']],"a:int,b:str")
        >>> ArrowDataFrame(schema = "a:int,b:int")  # empty dataframe
        >>> ArrowDataFrame(pd.DataFrame([[0]],columns=["a"]))
        >>> ArrowDataFrame(ArrayDataFrame([[0]],"a:int).as_arrow())
    """

    def __init__(self, df: Any=None, schema: Any=None):
        if df is None:
            schema = _input_schema(schema).assert_not_empty()
            self._native: pa.Table = schema.create_empty_arrow_table()
            super().__init__(schema)
            return
        elif isinstance(df, pa.Table):
            assert_or_throw(schema is None, InvalidOperationError("can't reset schema for pa.Table"))
            self._native = df
            super().__init__(Schema(df.schema))
            return
        elif isinstance(df, (pd.DataFrame, pd.Series)):
            if isinstance(df, pd.Series):
                df = df.to_frame()
            pdf = df
            if schema is None:
                self._native = pa.Table.from_pandas(pdf, schema=Schema(pdf).pa_schema, preserve_index=False, safe=True)
                schema = Schema(self._native.schema)
            else:
                schema = _input_schema(schema).assert_not_empty()
                if pdf.shape[0] == 0:
                    self._native = schema.create_empty_arrow_table()
                else:
                    self._native = pa.Table.from_pandas(pdf, schema=schema.pa_schema, preserve_index=False, safe=True)
            super().__init__(schema)
            return
        elif isinstance(df, Iterable):
            schema = _input_schema(schema).assert_not_empty()
            pdf = pd.DataFrame(df, columns=schema.names)
            if pdf.shape[0] == 0:
                self._native = schema.create_empty_arrow_table()
            else:
                for f in schema.fields:
                    if pa.types.is_timestamp(f.type) or pa.types.is_date(f.type):
                        pdf[f.name] = pd.to_datetime(pdf[f.name])
                schema = _input_schema(schema).assert_not_empty()
                self._native = pa.Table.from_pandas(pdf, schema=schema.pa_schema, preserve_index=False, safe=True)
            super().__init__(schema)
            return
        else:
            raise ValueError(f'{df} is incompatible with ArrowDataFrame')

    @property
    def native(self) -> pa.Table:
        """:func:`pyarrow.Table <pa:pyarrow.table>`"""
        return self._native

    def native_as_df(self) -> pa.Table:
        return self._native

    @property
    def empty(self) -> bool:
        return self.count() == 0

    def peek_array(self) -> List[Any]:
        self.assert_not_empty()
        data = self.native.take([0]).to_pydict()
        return [v[0] for v in data.values()]

    def peek_dict(self) -> Dict[str, Any]:
        self.assert_not_empty()
        data = self.native.take([0]).to_pydict()
        return {k: v[0] for k, v in data.items()}

    def count(self) -> int:
        return self.native.shape[0]

    def as_pandas(self) -> pd.DataFrame:
        return _pa_table_as_pandas(self.native)

    def head(self, n: int, columns: Optional[List[str]]=None) -> LocalBoundedDataFrame:
        adf = self.native if columns is None else self.native.select(columns)
        n = min(n, self.count())
        if n == 0:
            schema = self.schema if columns is None else self.schema.extract(columns)
            return ArrowDataFrame(None, schema=schema)
        return ArrowDataFrame(adf.take(list(range(n))))

    def _drop_cols(self, cols: List[str]) -> DataFrame:
        return ArrowDataFrame(self.native.drop(cols))

    def _select_cols(self, keys: List[Any]) -> DataFrame:
        return ArrowDataFrame(self.native.select(keys))

    def rename(self, columns: Dict[str, str]) -> DataFrame:
        try:
            cols = dict(columns)
            new_cols = [cols.pop(c, c) for c in self.columns]
            assert_or_throw(len(cols) == 0)
        except Exception as e:
            raise FugueDataFrameOperationError from e
        return ArrowDataFrame(self.native.rename_columns(new_cols))

    def alter_columns(self, columns: Any) -> DataFrame:
        adf = _pa_table_alter_columns(self.native, columns)
        if adf is self.native:
            return self
        return ArrowDataFrame(adf)

    def as_arrow(self, type_safe: bool=False) -> pa.Table:
        return self.native

    def as_array(self, columns: Optional[List[str]]=None, type_safe: bool=False) -> List[Any]:
        return pa_table_as_array(self.native, columns=columns)

    def as_dicts(self, columns: Optional[List[str]]=None) -> List[Dict[str, Any]]:
        return pa_table_as_dicts(self.native, columns=columns)

    def as_array_iterable(self, columns: Optional[List[str]]=None, type_safe: bool=False) -> Iterable[Any]:
        yield from pa_table_as_array_iterable(self.native, columns=columns)

    def as_dict_iterable(self, columns: Optional[List[str]]=None) -> Iterable[Dict[str, Any]]:
        yield from pa_table_as_dict_iterable(self.native, columns=columns)