from __future__ import annotations
import hashlib
import json
from typing import TYPE_CHECKING, Any, Final, Mapping, cast
from streamlit import config
from streamlit.proto.DeckGlJsonChart_pb2 import DeckGlJsonChart as PydeckProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
def _get_pydeck_tooltip(pydeck_obj: Deck | None) -> dict[str, str] | None:
    if pydeck_obj is None:
        return None
    desk_widget = getattr(pydeck_obj, 'deck_widget', None)
    if desk_widget is not None and isinstance(desk_widget.tooltip, dict):
        return desk_widget.tooltip
    tooltip = getattr(pydeck_obj, '_tooltip', None)
    if tooltip is not None and isinstance(tooltip, dict):
        return tooltip
    return None