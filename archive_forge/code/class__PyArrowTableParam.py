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
@fugue_annotated_param(pa.Table)
class _PyArrowTableParam(LocalDataFrameParam):

    def to_input_data(self, df: DataFrame, ctx: Any) -> Any:
        return df.as_arrow()

    def to_output_df(self, output: Any, schema: Any, ctx: Any) -> DataFrame:
        assert isinstance(output, pa.Table)
        adf: DataFrame = ArrowDataFrame(output)
        if schema is not None:
            _schema = Schema(schema)
            if adf.schema != _schema:
                adf = adf[_schema.names].alter_columns(_schema)
        return adf

    def count(self, df: Any) -> int:
        return df.count()

    def format_hint(self) -> Optional[str]:
        return 'pyarrow'