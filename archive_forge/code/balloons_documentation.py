from __future__ import annotations
from typing import TYPE_CHECKING, cast
from streamlit.proto.Balloons_pb2 import Balloons as BalloonsProto
from streamlit.runtime.metrics_util import gather_metrics
Get our DeltaGenerator.