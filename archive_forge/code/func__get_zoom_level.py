from __future__ import annotations
import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any, Collection, Dict, Final, Iterable, Union, cast
from typing_extensions import TypeAlias
import streamlit.elements.deck_gl_json_chart as deck_gl_json_chart
from streamlit import config, type_util
from streamlit.color_util import Color, IntColorTuple, is_color_like, to_int_color_tuple
from streamlit.errors import StreamlitAPIException
from streamlit.proto.DeckGlJsonChart_pb2 import DeckGlJsonChart as DeckGlJsonChartProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
def _get_zoom_level(distance: float) -> int:
    """Get the zoom level for a given distance in degrees.

    See https://wiki.openstreetmap.org/wiki/Zoom_levels for reference.

    Parameters
    ----------
    distance : float
        How many degrees of longitude should fit in the map.

    Returns
    -------
    int
        The zoom level, from 0 to 20.

    """
    for i in range(len(_ZOOM_LEVELS) - 1):
        if _ZOOM_LEVELS[i + 1] < distance <= _ZOOM_LEVELS[i]:
            return i
    return _DEFAULT_ZOOM_LEVEL