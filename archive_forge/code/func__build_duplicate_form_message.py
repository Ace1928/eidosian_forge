from __future__ import annotations
import textwrap
from typing import TYPE_CHECKING, Literal, NamedTuple, cast
from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.proto import Block_pb2
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import WidgetArgs, WidgetCallback, WidgetKwargs
def _build_duplicate_form_message(user_key: str | None=None) -> str:
    if user_key is not None:
        message = textwrap.dedent(f"\n            There are multiple identical forms with `key='{user_key}'`.\n\n            To fix this, please make sure that the `key` argument is unique for\n            each `st.form` you create.\n            ")
    else:
        message = textwrap.dedent("\n            There are multiple identical forms with the same generated key.\n\n            When a form is created, it's assigned an internal key based on\n            its structure. Multiple forms with an identical structure will\n            result in the same internal key, which causes this error.\n\n            To fix this error, please pass a unique `key` argument to\n            `st.form`.\n            ")
    return message.strip('\n')