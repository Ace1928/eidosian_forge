from __future__ import annotations
import textwrap
from typing import TYPE_CHECKING, Literal, NamedTuple, cast
from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.proto import Block_pb2
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs
def _current_form(this_dg: DeltaGenerator) -> FormData | None:
    """Find the FormData for the given DeltaGenerator.

    Forms are blocks, and can have other blocks nested inside them.
    To find the current form, we walk up the dg_stack until we find
    a DeltaGenerator that has FormData.
    """
    from streamlit.delta_generator import dg_stack
    if not runtime.exists():
        return None
    if this_dg._form_data is not None:
        return this_dg._form_data
    if this_dg == this_dg._main_dg:
        for dg in reversed(dg_stack.get()):
            if dg._form_data is not None:
                return dg._form_data
    else:
        parent = this_dg._parent
        if parent is not None and parent._form_data is not None:
            return parent._form_data
    return None