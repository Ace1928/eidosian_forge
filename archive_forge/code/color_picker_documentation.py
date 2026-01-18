from __future__ import annotations
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import cast
import streamlit
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ColorPicker_pb2 import ColorPicker as ColorPickerProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
Get our DeltaGenerator.