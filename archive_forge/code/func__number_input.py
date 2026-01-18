from __future__ import annotations
import numbers
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, Union, cast, overload
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber, JSNumberBoundsException
from streamlit.proto.NumberInput_pb2 import NumberInput as NumberInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
def _number_input(self, label: str, min_value: Number | None=None, max_value: Number | None=None, value: Number | Literal['min'] | None='min', step: Number | None=None, format: str | None=None, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, placeholder: str | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> Number | None:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=value if value != 'min' else None, key=key)
    maybe_raise_label_warnings(label, label_visibility)
    id = compute_widget_id('number_input', user_key=key, label=label, min_value=min_value, max_value=max_value, value=value, step=step, format=format, key=key, help=help, placeholder=None if placeholder is None else str(placeholder), form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    number_input_args = [min_value, max_value, value, step]
    int_args = all((isinstance(a, (numbers.Integral, type(None), str)) for a in number_input_args))
    float_args = all((isinstance(a, (float, type(None), str)) for a in number_input_args))
    if not int_args and (not float_args):
        raise StreamlitAPIException(f'All numerical arguments must be of the same type.\n`value` has {type(value).__name__} type.\n`min_value` has {type(min_value).__name__} type.\n`max_value` has {type(max_value).__name__} type.\n`step` has {type(step).__name__} type.')
    if value == 'min':
        if min_value is not None:
            value = min_value
        elif int_args and float_args:
            value = 0.0
        elif int_args:
            value = 0
        else:
            value = 0.0
    int_value = isinstance(value, numbers.Integral)
    float_value = isinstance(value, float)
    if value is None:
        if int_args and (not float_args):
            int_value = True
        else:
            float_value = True
    if format is None:
        format = '%d' if int_value else '%0.2f'
    if format in ['%d', '%u', '%i'] and float_value:
        import streamlit as st
        st.warning(f'Warning: NumberInput value below has type float, but format {format} displays as integer.')
    elif format[-1] == 'f' and int_value:
        import streamlit as st
        st.warning(f'Warning: NumberInput value below has type int so is displayed as int despite format string {format}.')
    if step is None:
        step = 1 if int_value else 0.01
    try:
        float(format % 2)
    except (TypeError, ValueError):
        raise StreamlitAPIException('Format string for st.number_input contains invalid characters: %s' % format)
    all_ints = int_value and int_args
    if min_value is not None and value is not None and (min_value > value):
        raise StreamlitAPIException(f'The default `value` {value} must be greater than or equal to the `min_value` {min_value}')
    if max_value is not None and value is not None and (max_value < value):
        raise StreamlitAPIException(f'The default `value` {value} must be less than or equal to the `max_value` {max_value}')
    try:
        if all_ints:
            if min_value is not None:
                JSNumber.validate_int_bounds(min_value, '`min_value`')
            if max_value is not None:
                JSNumber.validate_int_bounds(max_value, '`max_value`')
            if step is not None:
                JSNumber.validate_int_bounds(step, '`step`')
            if value is not None:
                JSNumber.validate_int_bounds(value, '`value`')
        else:
            if min_value is not None:
                JSNumber.validate_float_bounds(min_value, '`min_value`')
            if max_value is not None:
                JSNumber.validate_float_bounds(max_value, '`max_value`')
            if step is not None:
                JSNumber.validate_float_bounds(step, '`step`')
            if value is not None:
                JSNumber.validate_float_bounds(value, '`value`')
    except JSNumberBoundsException as e:
        raise StreamlitAPIException(str(e))
    data_type = NumberInputProto.INT if all_ints else NumberInputProto.FLOAT
    number_input_proto = NumberInputProto()
    number_input_proto.id = id
    number_input_proto.data_type = data_type
    number_input_proto.label = label
    if value is not None:
        number_input_proto.default = value
    if placeholder is not None:
        number_input_proto.placeholder = str(placeholder)
    number_input_proto.form_id = current_form_id(self.dg)
    number_input_proto.disabled = disabled
    number_input_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if help is not None:
        number_input_proto.help = dedent(help)
    if min_value is not None:
        number_input_proto.min = min_value
        number_input_proto.has_min = True
    if max_value is not None:
        number_input_proto.max = max_value
        number_input_proto.has_max = True
    if step is not None:
        number_input_proto.step = step
    if format is not None:
        number_input_proto.format = format
    serde = NumberInputSerde(value, data_type)
    widget_state = register_widget('number_input', number_input_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    if widget_state.value_changed:
        if widget_state.value is not None:
            number_input_proto.value = widget_state.value
        number_input_proto.set_value = True
    self.dg._enqueue('number_input', number_input_proto)
    return widget_state.value