from __future__ import annotations
import json
from enum import Enum
from typing import TYPE_CHECKING, Dict, Final, Literal, Mapping, Union
from typing_extensions import TypeAlias
from streamlit.elements.lib.column_types import ColumnConfig, ColumnType
from streamlit.elements.lib.dicttools import remove_none_values
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.type_util import DataFormat, is_colum_type_arrow_incompatible
def _determine_data_kind_via_pandas_dtype(column: Series | Index) -> ColumnDataKind:
    """Determine the data kind by using the pandas dtype.

    The column data kind refers to the shared data type of the values
    in the column (e.g. int, float, str, bool).

    Parameters
    ----------
    column : pd.Series, pd.Index
        The column for which the data kind should be determined.

    Returns
    -------
    ColumnDataKind
        The data kind of the column.
    """
    import pandas as pd
    column_dtype = column.dtype
    if pd.api.types.is_bool_dtype(column_dtype):
        return ColumnDataKind.BOOLEAN
    if pd.api.types.is_integer_dtype(column_dtype):
        return ColumnDataKind.INTEGER
    if pd.api.types.is_float_dtype(column_dtype):
        return ColumnDataKind.FLOAT
    if pd.api.types.is_datetime64_any_dtype(column_dtype):
        return ColumnDataKind.DATETIME
    if pd.api.types.is_timedelta64_dtype(column_dtype):
        return ColumnDataKind.TIMEDELTA
    if isinstance(column_dtype, pd.PeriodDtype):
        return ColumnDataKind.PERIOD
    if isinstance(column_dtype, pd.IntervalDtype):
        return ColumnDataKind.INTERVAL
    if pd.api.types.is_complex_dtype(column_dtype):
        return ColumnDataKind.COMPLEX
    if pd.api.types.is_object_dtype(column_dtype) is False and pd.api.types.is_string_dtype(column_dtype):
        return ColumnDataKind.STRING
    return ColumnDataKind.UNKNOWN