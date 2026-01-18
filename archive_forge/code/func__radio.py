from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, cast
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Radio_pb2 import Radio as RadioProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.type_util import (
from streamlit.util import index_
def _radio(self, label: str, options: OptionSequence[T], index: int | None=0, format_func: Callable[[Any], Any]=str, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, horizontal: bool=False, label_visibility: LabelVisibility='visible', captions: Sequence[str] | None=None, ctx: ScriptRunContext | None) -> T | None:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=None if index == 0 else index, key=key)
    maybe_raise_label_warnings(label, label_visibility)
    opt = ensure_indexable(options)
    check_python_comparable(opt)
    id = compute_widget_id('radio', user_key=key, label=label, options=[str(format_func(option)) for option in opt], index=index, key=key, help=help, horizontal=horizontal, captions=captions, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    if not isinstance(index, int) and index is not None:
        raise StreamlitAPIException('Radio Value has invalid type: %s' % type(index).__name__)
    if index is not None and len(opt) > 0 and (not 0 <= index < len(opt)):
        raise StreamlitAPIException('Radio index must be between 0 and length of options')

    def handle_captions(caption: str | None) -> str:
        if caption is None:
            return ''
        elif isinstance(caption, str):
            return caption
        else:
            raise StreamlitAPIException(f'Radio captions must be strings. Passed type: {type(caption).__name__}')
    radio_proto = RadioProto()
    radio_proto.id = id
    radio_proto.label = label
    if index is not None:
        radio_proto.default = index
    radio_proto.options[:] = [str(format_func(option)) for option in opt]
    radio_proto.form_id = current_form_id(self.dg)
    radio_proto.horizontal = horizontal
    radio_proto.disabled = disabled
    radio_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if captions is not None:
        radio_proto.captions[:] = map(handle_captions, captions)
    if help is not None:
        radio_proto.help = dedent(help)
    serde = RadioSerde(opt, index)
    widget_state = register_widget('radio', radio_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    widget_state = maybe_coerce_enum(widget_state, options, opt)
    if widget_state.value_changed:
        if widget_state.value is not None:
            serialized_value = serde.serialize(widget_state.value)
            if serialized_value is not None:
                radio_proto.value = serialized_value
        radio_proto.set_value = True
    if ctx:
        save_for_app_testing(ctx, id, format_func)
    self.dg._enqueue('radio', radio_proto)
    return widget_state.value