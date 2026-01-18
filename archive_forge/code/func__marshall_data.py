from __future__ import annotations
from typing import TYPE_CHECKING, Any
from streamlit import type_util
from streamlit.elements.lib import pandas_styler_utils
from streamlit.proto.Components_pb2 import ArrowTable as ArrowTableProto
def _marshall_data(proto: ArrowTableProto, df: DataFrame) -> None:
    """Marshall pandas.DataFrame data into an ArrowTable proto.

    Parameters
    ----------
    proto : proto.ArrowTable
        Output. The protobuf for a Streamlit ArrowTable proto.

    df : pandas.DataFrame
        A dataframe to marshall.

    """
    proto.data = type_util.data_frame_to_bytes(df)