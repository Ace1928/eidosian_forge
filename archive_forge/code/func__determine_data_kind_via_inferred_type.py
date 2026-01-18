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
def _determine_data_kind_via_inferred_type(column: Series | Index) -> ColumnDataKind:
    """Determine the data kind by inferring it from the underlying data.

    The column data kind refers to the shared data type of the values
    in the column (e.g. int, float, str, bool).

    Parameters
    ----------
    column : pd.Series, pd.Index
        The column to determine the data kind for.

    Returns
    -------
    ColumnDataKind
        The data kind of the column.
    """
    from pandas.api.types import infer_dtype
    inferred_type = infer_dtype(column)
    if inferred_type == 'string':
        return ColumnDataKind.STRING
    if inferred_type == 'bytes':
        return ColumnDataKind.BYTES
    if inferred_type in ['floating', 'mixed-integer-float']:
        return ColumnDataKind.FLOAT
    if inferred_type == 'integer':
        return ColumnDataKind.INTEGER
    if inferred_type == 'decimal':
        return ColumnDataKind.DECIMAL
    if inferred_type == 'complex':
        return ColumnDataKind.COMPLEX
    if inferred_type == 'boolean':
        return ColumnDataKind.BOOLEAN
    if inferred_type in ['datetime64', 'datetime']:
        return ColumnDataKind.DATETIME
    if inferred_type == 'date':
        return ColumnDataKind.DATE
    if inferred_type in ['timedelta64', 'timedelta']:
        return ColumnDataKind.TIMEDELTA
    if inferred_type == 'time':
        return ColumnDataKind.TIME
    if inferred_type == 'period':
        return ColumnDataKind.PERIOD
    if inferred_type == 'interval':
        return ColumnDataKind.INTERVAL
    if inferred_type == 'empty':
        return ColumnDataKind.EMPTY
    return ColumnDataKind.UNKNOWN