import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union
import dask.dataframe as dd
import pandas as pd
from distributed import Client
from triad.collections import Schema
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.pandas_like import PandasUtils
from triad.utils.threading import RunOnce
from triad.utils.io import makedirs
from fugue import StructuredRawSQL
from fugue.collections.partition import (
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue.dataframe import (
from fugue.dataframe.utils import get_join_schemas
from fugue.exceptions import FugueBug
from fugue.execution.execution_engine import ExecutionEngine, MapEngine, SQLEngine
from fugue.execution.native_execution_engine import NativeExecutionEngine
from fugue_dask._constants import FUGUE_DASK_DEFAULT_CONF
from fugue_dask._io import load_df, save_df
from fugue_dask._utils import (
from fugue_dask.dataframe import DaskDataFrame
from ._constants import FUGUE_DASK_USE_ARROW
class DaskMapEngine(MapEngine):

    @property
    def execution_engine_constraint(self) -> Type[ExecutionEngine]:
        return DaskExecutionEngine

    @property
    def is_distributed(self) -> bool:
        return True

    def map_dataframe(self, df: DataFrame, map_func: Callable[[PartitionCursor, LocalDataFrame], LocalDataFrame], output_schema: Any, partition_spec: PartitionSpec, on_init: Optional[Callable[[int, DataFrame], Any]]=None, map_func_format_hint: Optional[str]=None) -> DataFrame:
        presort = partition_spec.get_sorts(df.schema, with_partition_keys=partition_spec.algo == 'coarse')
        presort_keys = list(presort.keys())
        presort_asc = list(presort.values())
        output_schema = Schema(output_schema)
        output_dtypes = output_schema.to_pandas_dtype(use_extension_types=True, use_arrow_dtype=FUGUE_DASK_USE_ARROW)
        input_schema = df.schema
        cursor = partition_spec.get_cursor(input_schema, 0)
        on_init_once: Any = None if on_init is None else RunOnce(on_init, lambda *args, **kwargs: to_uuid(id(on_init), id(args[0])))

        def _fix_dask_bug(pdf: pd.DataFrame) -> pd.DataFrame:
            assert_or_throw(pdf.shape[1] == len(input_schema), FugueBug(f'partitioned dataframe has different number of columns: {pdf.columns} vs {input_schema}'))
            return pdf

        def _core_map(pdf: pd.DataFrame) -> pd.DataFrame:
            if len(partition_spec.presort) > 0:
                pdf = pdf.sort_values(presort_keys, ascending=presort_asc)
            input_df = PandasDataFrame(pdf.reset_index(drop=True), input_schema, pandas_df_wrapper=True)
            if on_init_once is not None:
                on_init_once(0, input_df)
            cursor.set(lambda: input_df.peek_array(), 0, 0)
            output_df = map_func(cursor, input_df)
            return output_df.as_pandas()[output_schema.names]

        def _map(pdf: pd.DataFrame) -> pd.DataFrame:
            if pdf.shape[0] == 0:
                return PandasDataFrame([], output_schema).as_pandas()
            pdf = pdf.reset_index(drop=True)
            pdf = _fix_dask_bug(pdf)
            res = _core_map(pdf)
            return res.astype(output_dtypes)

        def _gp_map(pdf: pd.DataFrame) -> pd.DataFrame:
            if pdf.shape[0] == 0:
                return PandasDataFrame([], output_schema).as_pandas()
            pdf = pdf.reset_index(drop=True)
            pdf = _fix_dask_bug(pdf)
            pu = PandasUtils()
            res = pu.safe_groupby_apply(pdf, partition_spec.partition_by, _core_map)
            return res.astype(output_dtypes)
        df = self.to_df(df)
        pdf = self.execution_engine.repartition(df, partition_spec)
        if len(partition_spec.partition_by) == 0:
            result = pdf.native.map_partitions(_map, meta=output_dtypes)
        elif partition_spec.algo == 'default':
            result = df.native.groupby(partition_spec.partition_by, sort=False, group_keys=False, dropna=False).apply(_map, meta=output_dtypes)
        elif partition_spec.algo == 'coarse':
            result = pdf.native.map_partitions(_map, meta=output_dtypes)
        else:
            result = pdf.native.map_partitions(_gp_map, meta=output_dtypes)
        return DaskDataFrame(result, output_schema)