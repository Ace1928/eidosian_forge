from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, Union, cast
from typing_extensions import TypeAlias
from streamlit.elements.utils import get_label_visibility_proto_value
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text
from streamlit.type_util import LabelVisibility, maybe_raise_label_warnings
def _determine_delta_color_and_direction(delta_color: DeltaColor, delta: Delta) -> MetricColorAndDirection:
    if delta_color not in {'normal', 'inverse', 'off'}:
        raise StreamlitAPIException(f"'{str(delta_color)}' is not an accepted value. delta_color only accepts: 'normal', 'inverse', or 'off'")
    if delta is None or delta == '':
        return MetricColorAndDirection(color=MetricProto.MetricColor.GRAY, direction=MetricProto.MetricDirection.NONE)
    if _is_negative_delta(delta):
        if delta_color == 'normal':
            cd_color = MetricProto.MetricColor.RED
        elif delta_color == 'inverse':
            cd_color = MetricProto.MetricColor.GREEN
        else:
            cd_color = MetricProto.MetricColor.GRAY
        cd_direction = MetricProto.MetricDirection.DOWN
    else:
        if delta_color == 'normal':
            cd_color = MetricProto.MetricColor.GREEN
        elif delta_color == 'inverse':
            cd_color = MetricProto.MetricColor.RED
        else:
            cd_color = MetricProto.MetricColor.GRAY
        cd_direction = MetricProto.MetricDirection.UP
    return MetricColorAndDirection(color=cd_color, direction=cd_direction)