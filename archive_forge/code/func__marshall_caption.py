from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypeVar
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
def _marshall_caption(proto: ArrowProto, styler: Styler) -> None:
    """Marshall pandas.Styler caption into an Arrow proto.

    Parameters
    ----------
    proto : proto.Arrow
        Output. The protobuf for Streamlit Arrow proto.

    styler : pandas.Styler
        Helps style a DataFrame or Series according to the data with HTML and CSS.

    """
    if styler.caption is not None:
        proto.styler.caption = styler.caption