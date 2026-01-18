import inspect
from typing import (
import pandas as pd
import pyarrow as pa
from triad import Schema, assert_or_throw
from triad.collections.function_wrapper import (
from triad.utils.iter import EmptyAwareIterable, make_empty_aware
from ..constants import FUGUE_ENTRYPOINT
from ..dataset.api import count as df_count
from .array_dataframe import ArrayDataFrame
from .arrow_dataframe import ArrowDataFrame
from .dataframe import AnyDataFrame, DataFrame, LocalDataFrame, as_fugue_df
from .dataframe_iterable_dataframe import (
from .dataframes import DataFrames
from .iterable_dataframe import IterableDataFrame
from .pandas_dataframe import PandasDataFrame
@fugue_annotated_param(AnyDataFrame)
class _AnyDataFrameParam(DataFrameParam):

    def to_output_df(self, output: AnyDataFrame, schema: Any, ctx: Any) -> DataFrame:
        return as_fugue_df(output) if schema is None else as_fugue_df(output, schema=schema)

    def count(self, df: Any) -> int:
        return df_count(df)