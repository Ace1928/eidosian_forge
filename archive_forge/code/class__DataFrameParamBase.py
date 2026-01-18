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
class _DataFrameParamBase(AnnotatedParam):

    def __init__(self, param: Optional[inspect.Parameter]):
        super().__init__(param)
        assert_or_throw(self.required, lambda: TypeError(f'{self} must be required'))

    def to_input_data(self, df: DataFrame, ctx: Any) -> Any:
        raise NotImplementedError

    def to_output_df(self, df: Any, schema: Any, ctx: Any) -> DataFrame:
        raise NotImplementedError

    def count(self, df: Any) -> int:
        raise NotImplementedError

    def need_schema(self) -> Optional[bool]:
        return False

    def format_hint(self) -> Optional[str]:
        return None