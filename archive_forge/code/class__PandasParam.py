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
@fugue_annotated_param(pd.DataFrame, 'p')
class _PandasParam(LocalDataFrameParam):

    @no_type_check
    def to_input_data(self, df: DataFrame, ctx: Any) -> pd.DataFrame:
        return df.as_pandas()

    @no_type_check
    def to_output_df(self, output: pd.DataFrame, schema: Any, ctx: Any) -> DataFrame:
        _schema: Optional[Schema] = None if schema is None else Schema(schema)
        if _schema is not None and _schema.names != list(output.columns):
            output = output[_schema.names]
        return PandasDataFrame(output, schema)

    @no_type_check
    def count(self, df: pd.DataFrame) -> int:
        return df.shape[0]

    def format_hint(self) -> Optional[str]:
        return 'pandas'