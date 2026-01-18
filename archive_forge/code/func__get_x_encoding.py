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
def _get_x_encoding(df: pd.DataFrame, x_column: str | None, x_from_user: str | None, chart_type: ChartType) -> alt.X:
    import altair as alt
    if x_column is None:
        x_field = NON_EXISTENT_COLUMN_NAME
        x_title = ''
    elif x_column == SEPARATED_INDEX_COLUMN_NAME:
        x_field = x_column
        x_title = ''
    else:
        x_field = x_column
        if x_from_user is None:
            x_title = ''
        else:
            x_title = x_column
    return alt.X(x_field, title=x_title, type=_get_x_encoding_type(df, chart_type, x_column), scale=_get_scale(df, x_column), axis=_get_axis_config(df, x_column, grid=False))