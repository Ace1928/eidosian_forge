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
class StreamlitColumnNotFoundError(StreamlitAPIException):

    def __init__(self, df, col_name, *args):
        available_columns = ', '.join((str(c) for c in list(df.columns)))
        message = f'Data does not have a column named `"{col_name}"`. Available columns are `{available_columns}`'
        super().__init__(message, *args)