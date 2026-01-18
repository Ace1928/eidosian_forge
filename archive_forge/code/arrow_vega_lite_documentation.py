from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Literal, cast
import streamlit.elements.lib.dicttools as dicttools
from streamlit.elements import arrow
from streamlit.elements.arrow import Data
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ArrowVegaLiteChart_pb2 import (
from streamlit.runtime.metrics_util import gather_metrics
Get our DeltaGenerator.