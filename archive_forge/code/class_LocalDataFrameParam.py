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
@fugue_annotated_param(LocalDataFrame, 'l', child_can_reuse_code=True)
class LocalDataFrameParam(DataFrameParam):

    def to_input_data(self, df: DataFrame, ctx: Any) -> LocalDataFrame:
        return df.as_local()

    def to_output_df(self, output: LocalDataFrame, schema: Any, ctx: Any) -> DataFrame:
        assert_or_throw(schema is None or output.schema == schema, lambda: f'Output schema mismatch {output.schema} vs {schema}')
        return output

    def count(self, df: LocalDataFrame) -> int:
        if df.is_bounded:
            return df.count()
        else:
            return sum((1 for _ in df.as_array_iterable()))