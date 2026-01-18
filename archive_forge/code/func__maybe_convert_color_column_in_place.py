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
def _maybe_convert_color_column_in_place(df: pd.DataFrame, color_column: str | None):
    """If needed, convert color column to a format Vega understands."""
    if color_column is None or len(df[color_column]) == 0:
        return
    first_color_datum = df[color_column].iat[0]
    if is_hex_color_like(first_color_datum):
        pass
    elif is_color_tuple_like(first_color_datum):
        df[color_column] = df[color_column].map(to_css_color)
    else:
        pass