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
def _generate_chart(chart_type: ChartType, data: Data | None, x_from_user: str | None=None, y_from_user: str | Sequence[str] | None=None, color_from_user: str | Color | list[Color] | None=None, size_from_user: str | float | None=None, width: int=0, height: int=0) -> tuple[alt.Chart, AddRowsMetadata]:
    """Function to use the chart's type, data columns and indices to figure out the chart's spec."""
    import altair as alt
    df = type_util.convert_anything_to_df(data, ensure_copy=True)
    del data
    x_column = _parse_x_column(df, x_from_user)
    y_column_list = _parse_y_columns(df, y_from_user, x_column)
    color_column, color_value = _parse_generic_column(df, color_from_user)
    size_column, size_value = _parse_generic_column(df, size_from_user)
    add_rows_metadata = AddRowsMetadata(last_index=last_index_for_melted_dataframes(df), columns=dict(x_column=x_column, y_column_list=y_column_list, color_column=color_column, size_column=size_column))
    df, x_column, y_column, color_column, size_column = prep_data(df, x_column, y_column_list, color_column, size_column)
    chart = alt.Chart(data=df, mark=chart_type.value['mark_type'], width=width, height=height).encode(x=_get_x_encoding(df, x_column, x_from_user, chart_type), y=_get_y_encoding(df, y_column, y_from_user))
    opacity_enc = _get_opacity_encoding(chart_type, color_column)
    if opacity_enc is not None:
        chart = chart.encode(opacity=opacity_enc)
    color_enc = _get_color_encoding(df, color_value, color_column, y_column_list, color_from_user)
    if color_enc is not None:
        chart = chart.encode(color=color_enc)
    size_enc = _get_size_encoding(chart_type, size_column, size_value)
    if size_enc is not None:
        chart = chart.encode(size=size_enc)
    if x_column is not None and y_column is not None:
        chart = chart.encode(tooltip=_get_tooltip_encoding(x_column, y_column, size_column, color_column, color_enc))
    return (chart.interactive(), add_rows_metadata)