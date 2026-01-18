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
def _parse_max_date(max_value: SingleDateValue, parsed_dates: Sequence[date] | None) -> date:
    parsed_max_date: date
    if isinstance(max_value, datetime):
        parsed_max_date = max_value.date()
    elif isinstance(max_value, date):
        parsed_max_date = max_value
    elif max_value is None:
        if parsed_dates:
            parsed_max_date = adjust_years(parsed_dates[-1], years=10)
        else:
            parsed_max_date = adjust_years(date.today(), years=10)
    else:
        raise StreamlitAPIException('DateInput max should either be a date/datetime or None')
    return parsed_max_date