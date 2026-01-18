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
def _convert_col_names_to_str_in_place(df: pd.DataFrame, x_column: str | None, y_column_list: list[str], color_column: str | None, size_column: str | None) -> tuple[str | None, list[str], str | None, str | None]:
    """Converts column names to strings, since Vega-Lite does not accept ints, etc."""
    import pandas as pd
    column_names = list(df.columns)
    str_column_names = [str(c) for c in column_names]
    df.columns = pd.Index(str_column_names)
    return (None if x_column is None else str(x_column), [str(c) for c in y_column_list], None if color_column is None else str(color_column), None if size_column is None else str(size_column))