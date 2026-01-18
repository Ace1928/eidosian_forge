from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, cast, overload
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.MultiSelect_pb2 import MultiSelect as MultiSelectProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.type_util import (
def _get_over_max_options_message(current_selections: int, max_selections: int):
    curr_selections_noun = 'option' if current_selections == 1 else 'options'
    max_selections_noun = 'option' if max_selections == 1 else 'options'
    return f"\nMultiselect has {current_selections} {curr_selections_noun} selected but `max_selections`\nis set to {max_selections}. This happened because you either gave too many options to `default`\nor you manipulated the widget's state through `st.session_state`. Note that\nthe latter can happen before the line indicated in the traceback.\nPlease select at most {max_selections} {max_selections_noun}.\n"