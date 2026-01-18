from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from numbers import Integral, Real
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Final, Sequence, Tuple, TypeVar, Union, cast
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber, JSNumberBoundsException
from streamlit.proto.Slider_pb2 import Slider as SliderProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
@dataclass
class SliderSerde:
    value: list[float]
    data_type: int
    single_value: bool
    orig_tz: tzinfo | None

    def deserialize(self, ui_value: list[float] | None, widget_id: str=''):
        if ui_value is not None:
            val: Any = ui_value
        else:
            val = self.value
        if self.data_type == SliderProto.INT:
            val = [int(v) for v in val]
        if self.data_type == SliderProto.DATETIME:
            val = [_micros_to_datetime(int(v), self.orig_tz) for v in val]
        if self.data_type == SliderProto.DATE:
            val = [_micros_to_datetime(int(v), self.orig_tz).date() for v in val]
        if self.data_type == SliderProto.TIME:
            val = [_micros_to_datetime(int(v), self.orig_tz).time().replace(tzinfo=self.orig_tz) for v in val]
        return val[0] if self.single_value else tuple(val)

    def serialize(self, v: Any) -> list[Any]:
        range_value = isinstance(v, (list, tuple))
        value = list(v) if range_value else [v]
        if self.data_type == SliderProto.DATE:
            value = [_datetime_to_micros(_date_to_datetime(v)) for v in value]
        if self.data_type == SliderProto.TIME:
            value = [_datetime_to_micros(_time_to_datetime(v)) for v in value]
        if self.data_type == SliderProto.DATETIME:
            value = [_datetime_to_micros(v) for v in value]
        return value