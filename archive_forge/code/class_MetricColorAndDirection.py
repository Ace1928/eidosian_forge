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
@dataclass(frozen=True)
class MetricColorAndDirection:
    color: MetricProto.MetricColor.ValueType
    direction: MetricProto.MetricDirection.ValueType