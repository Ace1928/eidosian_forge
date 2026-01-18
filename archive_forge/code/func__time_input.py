from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from textwrap import dedent
from typing import (
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.DateInput_pb2 import DateInput as DateInputProto
from streamlit.proto.TimeInput_pb2 import TimeInput as TimeInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.time_util import adjust_years
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
def _time_input(self, label: str, value: time | datetime | Literal['now'] | None='now', key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', step: int | timedelta=timedelta(minutes=DEFAULT_STEP_MINUTES), ctx: ScriptRunContext | None=None) -> time | None:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=value if value != 'now' else None, key=key)
    maybe_raise_label_warnings(label, label_visibility)
    parsed_time: time | None
    if value is None:
        parsed_time = None
    elif value == 'now':
        parsed_time = datetime.now().time().replace(second=0, microsecond=0)
    elif isinstance(value, datetime):
        parsed_time = value.time().replace(second=0, microsecond=0)
    elif isinstance(value, time):
        parsed_time = value
    else:
        raise StreamlitAPIException('The type of value should be one of datetime, time or None')
    id = compute_widget_id('time_input', user_key=key, label=label, value=parsed_time if isinstance(value, (datetime, time)) else value, key=key, help=help, step=step, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    del value
    time_input_proto = TimeInputProto()
    time_input_proto.id = id
    time_input_proto.label = label
    if parsed_time is not None:
        time_input_proto.default = time.strftime(parsed_time, '%H:%M')
    time_input_proto.form_id = current_form_id(self.dg)
    if not isinstance(step, (int, timedelta)):
        raise StreamlitAPIException(f'`step` can only be `int` or `timedelta` but {type(step)} is provided.')
    if isinstance(step, timedelta):
        step = step.seconds
    if step < 60 or step > timedelta(hours=23).seconds:
        raise StreamlitAPIException(f'`step` must be between 60 seconds and 23 hours but is currently set to {step} seconds.')
    time_input_proto.step = step
    time_input_proto.disabled = disabled
    time_input_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if help is not None:
        time_input_proto.help = dedent(help)
    serde = TimeInputSerde(parsed_time)
    widget_state = register_widget('time_input', time_input_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    if widget_state.value_changed:
        if (serialized_value := serde.serialize(widget_state.value)) is not None:
            time_input_proto.value = serialized_value
        time_input_proto.set_value = True
    self.dg._enqueue('time_input', time_input_proto)
    return widget_state.value