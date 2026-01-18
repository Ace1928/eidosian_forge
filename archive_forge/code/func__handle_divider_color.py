from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Literal, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Heading_pb2 import Heading as HeadingProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text
from streamlit.type_util import SupportsStr
@staticmethod
def _handle_divider_color(divider: Divider) -> str:
    if divider is True:
        return 'auto'
    valid_colors = ['blue', 'green', 'orange', 'red', 'violet', 'gray', 'grey', 'rainbow']
    if divider in valid_colors:
        return cast(str, divider)
    else:
        raise StreamlitAPIException(f'Divider parameter has invalid value: `{divider}`. Please choose from: {', '.join(valid_colors)}.')