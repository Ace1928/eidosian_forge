from __future__ import annotations
import io
import os
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, BinaryIO, Final, Literal, TextIO, Union, cast
from typing_extensions import TypeAlias
from streamlit import runtime, source_util
from streamlit.elements.form import current_form_id, is_in_form
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.file_util import get_main_script_directory, normalize_path_join
from streamlit.proto.Button_pb2 import Button as ButtonProto
from streamlit.proto.DownloadButton_pb2 import DownloadButton as DownloadButtonProto
from streamlit.proto.LinkButton_pb2 import LinkButton as LinkButtonProto
from streamlit.proto.PageLink_pb2 import PageLink as PageLinkProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.string_util import validate_emoji
from streamlit.type_util import Key, to_key
def _button(self, label: str, key: str | None, help: str | None, is_form_submitter: bool, on_click: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, type: Literal['primary', 'secondary']='secondary', disabled: bool=False, use_container_width: bool=False, ctx: ScriptRunContext | None=None) -> bool:
    if not is_form_submitter:
        check_callback_rules(self.dg, on_click)
    check_session_state_rules(default_value=None, key=key, writes_allowed=False)
    id = compute_widget_id('button', user_key=key, label=label, key=key, help=help, is_form_submitter=is_form_submitter, type=type, use_container_width=use_container_width, page=ctx.page_script_hash if ctx else None)
    if runtime.exists():
        if is_in_form(self.dg) and (not is_form_submitter):
            raise StreamlitAPIException(f"`st.button()` can't be used in an `st.form()`.{FORM_DOCS_INFO}")
        elif not is_in_form(self.dg) and is_form_submitter:
            raise StreamlitAPIException(f'`st.form_submit_button()` must be used inside an `st.form()`.{FORM_DOCS_INFO}')
    button_proto = ButtonProto()
    button_proto.id = id
    button_proto.label = label
    button_proto.default = False
    button_proto.is_form_submitter = is_form_submitter
    button_proto.form_id = current_form_id(self.dg)
    button_proto.type = type
    button_proto.use_container_width = use_container_width
    button_proto.disabled = disabled
    if help is not None:
        button_proto.help = dedent(help)
    serde = ButtonSerde()
    button_state = register_widget('button', button_proto, user_key=key, on_change_handler=on_click, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    if ctx:
        save_for_app_testing(ctx, id, button_state.value)
    self.dg._enqueue('button', button_proto)
    return button_state.value