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
def _maybe_reset_index_in_place(df: pd.DataFrame, x_column: str | None, y_column_list: list[str]) -> str | None:
    if x_column is None and len(y_column_list) > 0:
        if df.index.name is None:
            x_column = SEPARATED_INDEX_COLUMN_NAME
        else:
            x_column = df.index.name
        df.index.name = x_column
        df.reset_index(inplace=True)
    return x_column