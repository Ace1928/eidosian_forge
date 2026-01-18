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
def _color_picker(self, label: str, value: str | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> str:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=value, key=key)
    maybe_raise_label_warnings(label, label_visibility)
    id = compute_widget_id('color_picker', user_key=key, label=label, value=str(value), key=key, help=help, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    if value is None:
        value = '#000000'
    if not isinstance(value, str):
        raise StreamlitAPIException("\n                Color Picker Value has invalid type: %s. Expects a hex string\n                like '#00FFAA' or '#000'.\n                " % type(value).__name__)
    match = re.match('^#(?:[0-9a-fA-F]{3}){1,2}$', value)
    if not match:
        raise StreamlitAPIException("\n                '%s' is not a valid hex code for colors. Valid ones are like\n                '#00FFAA' or '#000'.\n                " % value)
    color_picker_proto = ColorPickerProto()
    color_picker_proto.id = id
    color_picker_proto.label = label
    color_picker_proto.default = str(value)
    color_picker_proto.form_id = current_form_id(self.dg)
    color_picker_proto.disabled = disabled
    color_picker_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if help is not None:
        color_picker_proto.help = dedent(help)
    serde = ColorPickerSerde(value)
    widget_state = register_widget('color_picker', color_picker_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    if widget_state.value_changed:
        color_picker_proto.value = widget_state.value
        color_picker_proto.set_value = True
    self.dg._enqueue('color_picker', color_picker_proto)
    return widget_state.value