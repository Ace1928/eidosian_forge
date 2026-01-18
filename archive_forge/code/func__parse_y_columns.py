from __future__ import annotations
from contextlib import nullcontext
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any, Collection, Literal, Sequence, cast
import streamlit.elements.arrow_vega_lite as arrow_vega_lite
from streamlit import type_util
from streamlit.color_util import (
from streamlit.elements.altair_utils import AddRowsMetadata
from streamlit.elements.arrow import Data
from streamlit.elements.utils import last_index_for_melted_dataframes
from streamlit.errors import Error, StreamlitAPIException
from streamlit.proto.ArrowVegaLiteChart_pb2 import (
from streamlit.runtime.metrics_util import gather_metrics
def _parse_y_columns(df: pd.DataFrame, y_from_user: str | Sequence[str] | None, x_column: str | None) -> list[str]:
    y_column_list: list[str] = []
    if y_from_user is None:
        y_column_list = list(df.columns)
    elif isinstance(y_from_user, str):
        y_column_list = [y_from_user]
    elif type_util.is_sequence(y_from_user):
        y_column_list = list((str(col) for col in y_from_user))
    else:
        raise StreamlitAPIException(f'y parameter should be a column name (str) or list thereof. Value given: {y_from_user} (type {type(y_from_user)})')
    for col in y_column_list:
        if col not in df.columns:
            raise StreamlitColumnNotFoundError(df, col)
    if x_column in y_column_list and (not y_from_user or x_column not in y_from_user):
        y_column_list.remove(x_column)
    return y_column_list