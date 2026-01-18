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
@dataclass(frozen=True)
class _DateInputValues:
    value: Sequence[date] | None
    is_range: bool
    max: date
    min: date

    @classmethod
    def from_raw_values(cls, value: DateValue | Literal['today'] | Literal['default_value_today'], min_value: SingleDateValue, max_value: SingleDateValue) -> _DateInputValues:
        parsed_value, is_range = _parse_date_value(value=value)
        return cls(value=parsed_value, is_range=is_range, min=_parse_min_date(min_value=min_value, parsed_dates=parsed_value), max=_parse_max_date(max_value=max_value, parsed_dates=parsed_value))

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise StreamlitAPIException(f"The `min_value`, set to {self.min}, shouldn't be larger than the `max_value`, set to {self.max}.")
        if self.value:
            start_value = self.value[0]
            end_value = self.value[-1]
            if start_value < self.min or end_value > self.max:
                raise StreamlitAPIException(f'The default `value` of {self.value} must lie between the `min_value` of {self.min} and the `max_value` of {self.max}, inclusively.')