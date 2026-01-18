import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union
import numpy as np
import pandas as pd
from triad import Schema
from triad.collections.dict import IndexedOrderedDict
from triad.utils.assertion import assert_or_throw
from triad.utils.io import makedirs
from triad.utils.pandas_like import PandasUtils
from fugue._utils.io import load_df, save_df
from fugue._utils.misc import import_fsql_dependency
from fugue.collections.partition import (
from fugue.collections.sql import StructuredRawSQL
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue.dataframe import (
from fugue.dataframe.dataframe import as_fugue_df
from fugue.dataframe.utils import get_join_schemas
from .execution_engine import (
class NativeExecutionEngine(ExecutionEngine):
    """The execution engine based on native python and pandas. This execution engine
    is mainly for prototyping and unit tests.

    Please read |ExecutionEngineTutorial| to understand this important Fugue concept

    :param conf: |ParamsLikeObject|, read |FugueConfig| to learn Fugue specific options
    """

    def __init__(self, conf: Any=None):
        super().__init__(conf)
        self._log = logging.getLogger()

    def __repr__(self) -> str:
        return 'NativeExecutionEngine'

    @property
    def log(self) -> logging.Logger:
        return self._log

    @property
    def is_distributed(self) -> bool:
        return False

    def create_default_sql_engine(self) -> SQLEngine:
        return QPDPandasEngine(self)

    def create_default_map_engine(self) -> MapEngine:
        return PandasMapEngine(self)

    def get_current_parallelism(self) -> int:
        return 1

    @property
    def pl_utils(self) -> PandasUtils:
        """Pandas-like dataframe utils"""
        return PandasUtils()

    def to_df(self, df: AnyDataFrame, schema: Any=None) -> LocalBoundedDataFrame:
        return _to_native_execution_engine_df(df, schema)

    def repartition(self, df: DataFrame, partition_spec: PartitionSpec) -> DataFrame:
        return df

    def broadcast(self, df: DataFrame) -> DataFrame:
        return self.to_df(df)

    def persist(self, df: DataFrame, lazy: bool=False, **kwargs: Any) -> DataFrame:
        return self.to_df(df)

    def join(self, df1: DataFrame, df2: DataFrame, how: str, on: Optional[List[str]]=None) -> DataFrame:
        key_schema, output_schema = get_join_schemas(df1, df2, how=how, on=on)
        d = self.pl_utils.join(df1.as_pandas(), df2.as_pandas(), join_type=how, on=key_schema.names)
        return PandasDataFrame(d.reset_index(drop=True), output_schema)

    def union(self, df1: DataFrame, df2: DataFrame, distinct: bool=True) -> DataFrame:
        assert_or_throw(df1.schema == df2.schema, lambda: ValueError(f'{df1.schema} != {df2.schema}'))
        d = self.pl_utils.union(df1.as_pandas(), df2.as_pandas(), unique=distinct)
        return PandasDataFrame(d.reset_index(drop=True), df1.schema)

    def subtract(self, df1: DataFrame, df2: DataFrame, distinct: bool=True) -> DataFrame:
        assert_or_throw(distinct, NotImplementedError('EXCEPT ALL for NativeExecutionEngine'))
        assert_or_throw(df1.schema == df2.schema, lambda: ValueError(f'{df1.schema} != {df2.schema}'))
        d = self.pl_utils.except_df(df1.as_pandas(), df2.as_pandas(), unique=distinct)
        return PandasDataFrame(d.reset_index(drop=True), df1.schema)

    def intersect(self, df1: DataFrame, df2: DataFrame, distinct: bool=True) -> DataFrame:
        assert_or_throw(distinct, NotImplementedError('INTERSECT ALL for NativeExecutionEngine'))
        assert_or_throw(df1.schema == df2.schema, lambda: ValueError(f'{df1.schema} != {df2.schema}'))
        d = self.pl_utils.intersect(df1.as_pandas(), df2.as_pandas(), unique=distinct)
        return PandasDataFrame(d.reset_index(drop=True), df1.schema)

    def distinct(self, df: DataFrame) -> DataFrame:
        d = self.pl_utils.drop_duplicates(df.as_pandas())
        return PandasDataFrame(d.reset_index(drop=True), df.schema)

    def dropna(self, df: DataFrame, how: str='any', thresh: Optional[int]=None, subset: List[str]=None) -> DataFrame:
        kwargs: Dict[str, Any] = dict(axis=0, subset=subset, inplace=False)
        if thresh is None:
            kwargs['how'] = how
        else:
            kwargs['thresh'] = thresh
        d = df.as_pandas().dropna(**kwargs)
        return PandasDataFrame(d.reset_index(drop=True), df.schema)

    def fillna(self, df: DataFrame, value: Any, subset: List[str]=None) -> DataFrame:
        assert_or_throw(not isinstance(value, list) and value is not None, ValueError('fillna value can not None or a list'))
        if isinstance(value, dict):
            assert_or_throw(None not in value.values() and any(value.values()), ValueError('fillna dict can not contain None and needs at least one value'))
            mapping = value
        else:
            subset = subset or df.columns
            mapping = {col: value for col in subset}
        d = df.as_pandas().fillna(mapping, inplace=False)
        return PandasDataFrame(d.reset_index(drop=True), df.schema)

    def sample(self, df: DataFrame, n: Optional[int]=None, frac: Optional[float]=None, replace: bool=False, seed: Optional[int]=None) -> DataFrame:
        assert_or_throw(n is None and frac is not None or (n is not None and frac is None), ValueError('one and only one of n and frac should be set'))
        d = df.as_pandas().sample(n=n, frac=frac, replace=replace, random_state=seed)
        return PandasDataFrame(d.reset_index(drop=True), df.schema)

    def take(self, df: DataFrame, n: int, presort: str, na_position: str='last', partition_spec: Optional[PartitionSpec]=None) -> DataFrame:
        partition_spec = partition_spec or PartitionSpec()
        assert_or_throw(isinstance(n, int), ValueError('n needs to be an integer'))
        d = df.as_pandas()
        if presort:
            presort = parse_presort_exp(presort)
        _presort: IndexedOrderedDict = presort or partition_spec.presort
        if len(_presort.keys()) > 0:
            d = d.sort_values(list(_presort.keys()), ascending=list(_presort.values()), na_position=na_position)
        if len(partition_spec.partition_by) == 0:
            d = d.head(n)
        else:
            d = d.groupby(by=partition_spec.partition_by, dropna=False).head(n)
        return PandasDataFrame(d.reset_index(drop=True), df.schema, pandas_df_wrapper=True)

    def load_df(self, path: Union[str, List[str]], format_hint: Any=None, columns: Any=None, **kwargs: Any) -> LocalBoundedDataFrame:
        return self.to_df(load_df(path, format_hint=format_hint, columns=columns, **kwargs))

    def save_df(self, df: DataFrame, path: str, format_hint: Any=None, mode: str='overwrite', partition_spec: Optional[PartitionSpec]=None, force_single: bool=False, **kwargs: Any) -> None:
        partition_spec = partition_spec or PartitionSpec()
        if not force_single and (not partition_spec.empty):
            kwargs['partition_cols'] = partition_spec.partition_by
        makedirs(os.path.dirname(path), exist_ok=True)
        df = self.to_df(df)
        save_df(df, path, format_hint=format_hint, mode=mode, **kwargs)