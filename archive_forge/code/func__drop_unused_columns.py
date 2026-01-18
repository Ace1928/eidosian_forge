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
def _drop_unused_columns(df: pd.DataFrame, *column_names: str | None) -> pd.DataFrame:
    """Returns a subset of df, selecting only column_names that aren't None."""
    seen = set()
    keep = []
    for x in column_names:
        if x is None:
            continue
        if x in seen:
            continue
        seen.add(x)
        keep.append(x)
    return df[keep]