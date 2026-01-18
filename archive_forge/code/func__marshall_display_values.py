from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypeVar
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
def _marshall_display_values(proto: ArrowProto, df: DataFrame, styles: Mapping[str, Any]) -> None:
    """Marshall pandas.Styler display values into an Arrow proto.

    Parameters
    ----------
    proto : proto.Arrow
        Output. The protobuf for Streamlit Arrow proto.

    df : pandas.DataFrame
        A dataframe with original values.

    styles : dict
        pandas.Styler translated styles.

    """
    new_df = _use_display_values(df, styles)
    proto.styler.display_values = type_util.data_frame_to_bytes(new_df)