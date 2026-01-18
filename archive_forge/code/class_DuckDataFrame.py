from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import pyarrow as pa
from duckdb import DuckDBPyRelation
from triad import Schema, assert_or_throw
from triad.utils.pyarrow import LARGE_TYPES_REPLACEMENT, replace_types_in_table
from fugue import ArrowDataFrame, DataFrame, LocalBoundedDataFrame
from fugue.dataframe.arrow_dataframe import _pa_table_as_pandas
from fugue.dataframe.utils import (
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
from fugue.plugins import (
from ._utils import encode_column_name, to_duck_type, to_pa_type
class DuckDataFrame(LocalBoundedDataFrame):
    """DataFrame that wraps DuckDB ``DuckDBPyRelation``.

    :param rel: ``DuckDBPyRelation`` object
    """

    def __init__(self, rel: DuckDBPyRelation):
        self._rel = rel
        super().__init__(schema=lambda: _duck_get_schema(self._rel))

    @property
    def alias(self) -> str:
        return '_' + str(id(self._rel))

    @property
    def native(self) -> DuckDBPyRelation:
        """DuckDB relation object"""
        return self._rel

    def native_as_df(self) -> DuckDBPyRelation:
        return self._rel

    @property
    def empty(self) -> bool:
        return self.count() == 0

    def peek_array(self) -> List[Any]:
        res = self._rel.limit(1).to_df()
        if res.shape[0] == 0:
            raise FugueDatasetEmptyError()
        return res.values.tolist()[0]

    def count(self) -> int:
        return len(self._rel)

    def _drop_cols(self, cols: List[str]) -> DataFrame:
        return DuckDataFrame(_drop_duckdb_columns(self._rel, cols))

    def _select_cols(self, keys: List[Any]) -> DataFrame:
        return DuckDataFrame(_select_duckdb_columns(self._rel, keys))

    def rename(self, columns: Dict[str, str]) -> DataFrame:
        _assert_no_missing(self._rel, columns.keys())
        expr = ', '.join((f'{a} AS {b}' for a, b in [(encode_column_name(name), encode_column_name(columns.get(name, name))) for name in self._rel.columns]))
        return DuckDataFrame(self._rel.project(expr))

    def alter_columns(self, columns: Any) -> DataFrame:
        new_schema = self.schema.alter(columns)
        if new_schema == self.schema:
            return self
        fields: List[str] = []
        for f1, f2 in zip(self.schema.fields, new_schema.fields):
            if f1.type == f2.type:
                fields.append(encode_column_name(f1.name))
            else:
                tp = to_duck_type(f2.type)
                fields.append(f'CAST({encode_column_name(f1.name)} AS {tp}) AS {encode_column_name(f1.name)}')
        return DuckDataFrame(self._rel.project(', '.join(fields)))

    def as_arrow(self, type_safe: bool=False) -> pa.Table:
        return _duck_as_arrow(self._rel)

    def as_pandas(self) -> pd.DataFrame:
        return _duck_as_pandas(self._rel)

    def as_local_bounded(self) -> LocalBoundedDataFrame:
        res = ArrowDataFrame(self.as_arrow())
        if self.has_metadata:
            res.reset_metadata(self.metadata)
        return res

    def as_array(self, columns: Optional[List[str]]=None, type_safe: bool=False) -> List[Any]:
        return _duck_as_array(self._rel, columns=columns, type_safe=type_safe)

    def as_array_iterable(self, columns: Optional[List[str]]=None, type_safe: bool=False) -> Iterable[Any]:
        yield from _duck_as_array_iterable(self._rel, columns=columns, type_safe=type_safe)

    def as_dicts(self, columns: Optional[List[str]]=None) -> List[Dict[str, Any]]:
        return _duck_as_dicts(self._rel, columns=columns)

    def as_dict_iterable(self, columns: Optional[List[str]]=None) -> Iterable[Dict[str, Any]]:
        yield from _duck_as_dict_iterable(self._rel, columns=columns)

    def head(self, n: int, columns: Optional[List[str]]=None) -> LocalBoundedDataFrame:
        if columns is not None:
            return self[columns].head(n)
        return ArrowDataFrame(_duck_as_arrow(self._rel.limit(n)))