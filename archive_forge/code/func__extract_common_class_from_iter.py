from __future__ import annotations
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Sequence, cast, overload
import streamlit
from streamlit import config, runtime, type_util
from streamlit.elements.form import is_in_form
from streamlit.errors import StreamlitAPIException
from streamlit.proto.LabelVisibilityMessage_pb2 import LabelVisibilityMessage
from streamlit.runtime.state import WidgetCallback, get_session_state
from streamlit.runtime.state.common import RegisterWidgetResult
from streamlit.type_util import T
def _extract_common_class_from_iter(iterable: Iterable[Any]) -> Any:
    """Return the common class of all elements in a iterable if they share one.
    Otherwise, return None."""
    try:
        inner_iter = iter(iterable)
        first_class = type(next(inner_iter))
    except StopIteration:
        return None
    if all((type(item) is first_class for item in inner_iter)):
        return first_class
    return None