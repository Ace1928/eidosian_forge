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
def _convert_color_arg_or_column(data: DataFrame, color_arg: str | Color, color_col_name: str | None) -> None | str | IntColorTuple:
    """Converts color to a format accepted by PyDeck.

    For example:
    - If color_arg is "#fff", then returns (255, 255, 255, 255).
    - If color_col_name is "my_col_123", then it converts everything in column my_col_123 to
      an accepted color format such as (0, 100, 200, 255).

    NOTE: This function mutates the data argument.
    """
    color_arg_out: None | str | IntColorTuple = None
    if color_col_name is not None:
        if len(data[color_col_name]) > 0 and is_color_like(data[color_col_name].iat[0]):
            data.loc[:, color_col_name] = data.loc[:, color_col_name].map(to_int_color_tuple)
        else:
            raise StreamlitAPIException(f'Column "{color_col_name}" does not appear to contain valid colors.')
        assert isinstance(color_arg, str)
        color_arg_out = color_arg
    elif color_arg is not None:
        color_arg_out = to_int_color_tuple(color_arg)
    return color_arg_out