from __future__ import annotations
import hashlib
import json
from typing import TYPE_CHECKING, Any, Final, Mapping, cast
from streamlit import config
from streamlit.proto.DeckGlJsonChart_pb2 import DeckGlJsonChart as PydeckProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
Get our DeltaGenerator.