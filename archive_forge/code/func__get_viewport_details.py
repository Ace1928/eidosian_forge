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
def _get_viewport_details(data: DataFrame, lat_col_name: str, lon_col_name: str, zoom: int | None) -> tuple[int, float, float]:
    """Auto-set viewport when not fully specified by user."""
    min_lat = data[lat_col_name].min()
    max_lat = data[lat_col_name].max()
    min_lon = data[lon_col_name].min()
    max_lon = data[lon_col_name].max()
    center_lat = (max_lat + min_lat) / 2.0
    center_lon = (max_lon + min_lon) / 2.0
    range_lon = abs(max_lon - min_lon)
    range_lat = abs(max_lat - min_lat)
    if zoom is None:
        if range_lon > range_lat:
            longitude_distance = range_lon
        else:
            longitude_distance = range_lat
        zoom = _get_zoom_level(longitude_distance)
    return (zoom, center_lat, center_lon)