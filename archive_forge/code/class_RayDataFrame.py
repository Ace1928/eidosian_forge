from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
import ray
import ray.data as rd
from triad import assert_or_throw
from triad.collections.schema import Schema
from triad.utils.pyarrow import cast_pa_table
from fugue.dataframe import ArrowDataFrame, DataFrame, LocalBoundedDataFrame
from fugue.dataframe.dataframe import _input_schema
from fugue.dataframe.utils import pa_table_as_array, pa_table_as_dicts
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
from fugue.plugins import (
from ._constants import _ZERO_COPY
from ._utils.dataframe import build_empty, get_dataset_format, materialize, to_schema
class RayDataFrame(DataFrame):
    """DataFrame that wraps Ray DataSet. Please also read
    |DataFrameTutorial| to understand this Fugue concept

    :param df: :class:`ray:ray.data.Dataset`, :class:`pa:pyarrow.Table`,
      :class:`pd:pandas.DataFrame`,
      Fugue :class:`~fugue.dataframe.dataframe.DataFrame`,
      or list or iterable of arrays
    :param schema: |SchemaLikeObject|, defaults to None. If the schema
      is different from the ``df`` schema, then type casts will happen.
    :param internal_schema: for internal schema, it means the schema
      is guaranteed by the provider to be consistent with the schema of
      ``df``, so no type cast will happen. Defaults to False. This is
      for internal use only.
    """

    def __init__(self, df: Any=None, schema: Any=None, internal_schema: bool=False):
        metadata: Any = None
        if internal_schema:
            schema = _input_schema(schema).assert_not_empty()
        if df is None:
            schema = _input_schema(schema).assert_not_empty()
            super().__init__(schema)
            self._native = build_empty(schema)
            return
        if isinstance(df, rd.Dataset):
            fmt, df = get_dataset_format(df)
            if fmt is None:
                schema = _input_schema(schema).assert_not_empty()
                super().__init__(schema)
                self._native = build_empty(schema)
                return
            elif fmt == 'pandas':
                rdf = rd.from_arrow_refs(df.to_arrow_refs())
            elif fmt == 'arrow':
                rdf = df
            else:
                raise NotImplementedError(f'Ray Dataset in {fmt} format is not supported')
        elif isinstance(df, pa.Table):
            rdf = rd.from_arrow(df)
            if schema is None:
                schema = df.schema
        elif isinstance(df, RayDataFrame):
            rdf = df._native
            if schema is None:
                schema = df.schema
            metadata = None if not df.has_metadata else df.metadata
        elif isinstance(df, (pd.DataFrame, pd.Series)):
            if isinstance(df, pd.Series):
                df = df.to_frame()
            adf = ArrowDataFrame(df)
            rdf = rd.from_arrow(adf.native)
            if schema is None:
                schema = adf.schema
        elif isinstance(df, Iterable):
            schema = _input_schema(schema).assert_not_empty()
            t = ArrowDataFrame(df, schema)
            rdf = rd.from_arrow(t.as_arrow())
        elif isinstance(df, DataFrame):
            rdf = rd.from_arrow(df.as_arrow(type_safe=True))
            if schema is None:
                schema = df.schema
            metadata = None if not df.has_metadata else df.metadata
        else:
            raise ValueError(f'{df} is incompatible with RayDataFrame')
        rdf, schema = self._apply_schema(rdf, schema, internal_schema)
        super().__init__(schema)
        self._native = rdf
        if metadata is not None:
            self.reset_metadata(metadata)

    @property
    def native(self) -> rd.Dataset:
        """The wrapped ray Dataset"""
        return self._native

    def native_as_df(self) -> rd.Dataset:
        return self._native

    @property
    def is_local(self) -> bool:
        return False

    def as_local_bounded(self) -> LocalBoundedDataFrame:
        adf = self.as_arrow()
        if adf.shape[0] == 0:
            res = ArrowDataFrame([], self.schema)
        else:
            res = ArrowDataFrame(adf)
        if self.has_metadata:
            res.reset_metadata(self.metadata)
        return res

    @property
    def is_bounded(self) -> bool:
        return True

    @property
    def empty(self) -> bool:
        return len(self.native.take(1)) == 0

    @property
    def num_partitions(self) -> int:
        return _rd_num_partitions(self.native)

    def _drop_cols(self, cols: List[str]) -> DataFrame:
        cols = (self.schema - cols).names
        return self._select_cols(cols)

    def _select_cols(self, cols: List[Any]) -> DataFrame:
        if cols == self.columns:
            return self
        return RayDataFrame(self.native.select_columns(cols), self.schema.extract(cols), internal_schema=True)

    def peek_array(self) -> List[Any]:
        data = self.native.limit(1).to_pandas().values.tolist()
        if len(data) == 0:
            raise FugueDatasetEmptyError
        return data[0]

    def persist(self, **kwargs: Any) -> 'RayDataFrame':
        self._native = materialize(self._native)
        return self

    def count(self) -> int:
        return self.native.count()

    def as_arrow(self, type_safe: bool=False) -> pa.Table:
        return _rd_as_arrow(self.native)

    def as_pandas(self) -> pd.DataFrame:
        return _rd_as_pandas(self.native)

    def rename(self, columns: Dict[str, str]) -> DataFrame:
        try:
            new_schema = self.schema.rename(columns)
            new_cols = new_schema.names
        except Exception as e:
            raise FugueDataFrameOperationError from e
        rdf = self.native.map_batches(lambda b: b.rename_columns(new_cols), batch_format='pyarrow', **_ZERO_COPY, **self._remote_args())
        return RayDataFrame(rdf, schema=new_schema, internal_schema=True)

    def alter_columns(self, columns: Any) -> DataFrame:
        new_schema = self.schema.alter(columns)

        def _alter(table: pa.Table) -> pa.Table:
            return cast_pa_table(table, new_schema.pa_schema)
        if self.schema == new_schema:
            return self
        rdf = self.native.map_batches(_alter, batch_format='pyarrow', **_ZERO_COPY, **self._remote_args())
        return RayDataFrame(rdf, schema=new_schema, internal_schema=True)

    def as_array(self, columns: Optional[List[str]]=None, type_safe: bool=False) -> List[Any]:
        return _rd_as_array(self.native, columns, type_safe)

    def as_array_iterable(self, columns: Optional[List[str]]=None, type_safe: bool=False) -> Iterable[Any]:
        yield from _rd_as_array_iterable(self.native, columns, type_safe)

    def as_dicts(self, columns: Optional[List[str]]=None) -> List[Dict[str, Any]]:
        return _rd_as_dicts(self.native, columns)

    def as_dict_iterable(self, columns: Optional[List[str]]=None) -> Iterable[Dict[str, Any]]:
        yield from _rd_as_dict_iterable(self.native, columns)

    def head(self, n: int, columns: Optional[List[str]]=None) -> LocalBoundedDataFrame:
        if columns is not None:
            return self[columns].head(n)
        pdf = RayDataFrame(self.native.limit(n), schema=self.schema)
        return pdf.as_local()

    def _apply_schema(self, rdf: rd.Dataset, schema: Optional[Schema], internal_schema: bool) -> Tuple[rd.Dataset, Schema]:
        if internal_schema:
            return (rdf, schema)
        fmt, rdf = get_dataset_format(rdf)
        if fmt is None:
            schema = _input_schema(schema).assert_not_empty()
            return (build_empty(schema), schema)
        if schema is None or schema == to_schema(rdf.schema(fetch_if_missing=True)):
            return (rdf, to_schema(rdf.schema(fetch_if_missing=True)))

        def _alter(table: pa.Table) -> pa.Table:
            return ArrowDataFrame(table).alter_columns(schema).native
        return (rdf.map_batches(_alter, batch_format='pyarrow', **_ZERO_COPY, **self._remote_args()), schema)

    def _remote_args(self) -> Dict[str, Any]:
        return {'num_cpus': 1}