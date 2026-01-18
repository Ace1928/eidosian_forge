from __future__ import annotations
import textwrap
from typing import TYPE_CHECKING, Literal, NamedTuple, cast
from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.proto import Block_pb2
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs
def _form_submit_button(self, label: str='Submit', help: str | None=None, on_click: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, type: Literal['primary', 'secondary']='secondary', disabled: bool=False, use_container_width: bool=False, ctx: ScriptRunContext | None=None) -> bool:
    form_id = current_form_id(self.dg)
    submit_button_key = f'FormSubmitter:{form_id}-{label}'
    return self.dg._button(label=label, key=submit_button_key, help=help, is_form_submitter=True, on_click=on_click, args=args, kwargs=kwargs, type=type, disabled=disabled, use_container_width=use_container_width, ctx=ctx)