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
def _parse_generic_column(df: pd.DataFrame, column_or_value: Any) -> tuple[str | None, Any]:
    if isinstance(column_or_value, str) and column_or_value in df.columns:
        column_name = column_or_value
        value = None
    else:
        column_name = None
        value = column_or_value
    return (column_name, value)