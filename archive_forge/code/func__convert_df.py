from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.utils.rename import normalize_names
from .._utils.registry import fugue_plugin
from .dataframe import AnyDataFrame, DataFrame, as_fugue_df
def _convert_df(input_df: AnyDataFrame, output_df: DataFrame, as_fugue: bool) -> AnyDataFrame:
    if as_fugue or isinstance(input_df, DataFrame):
        return output_df
    return output_df.native_as_df()