from __future__ import annotations
from typing import TYPE_CHECKING, Any
from streamlit import type_util
from streamlit.elements.lib import pandas_styler_utils
from streamlit.proto.Components_pb2 import ArrowTable as ArrowTableProto
Convert ArrowTable proto to pandas.DataFrame.

    Parameters
    ----------
    proto : proto.ArrowTable
        Output. pandas.DataFrame

    