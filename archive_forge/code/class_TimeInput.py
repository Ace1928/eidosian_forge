from __future__ import annotations
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, time, timedelta
from typing import (
from typing_extensions import TypeAlias
from streamlit import type_util, util
from streamlit.elements.heading import HeadingProtoTag
from streamlit.elements.widgets.select_slider import SelectSliderSerde
from streamlit.elements.widgets.slider import (
from streamlit.elements.widgets.time_widgets import (
from streamlit.proto.Alert_pb2 import Alert as AlertProto
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.proto.Button_pb2 import Button as ButtonProto
from streamlit.proto.ChatInput_pb2 import ChatInput as ChatInputProto
from streamlit.proto.Checkbox_pb2 import Checkbox as CheckboxProto
from streamlit.proto.Code_pb2 import Code as CodeProto
from streamlit.proto.ColorPicker_pb2 import ColorPicker as ColorPickerProto
from streamlit.proto.DateInput_pb2 import DateInput as DateInputProto
from streamlit.proto.Element_pb2 import Element as ElementProto
from streamlit.proto.Exception_pb2 import Exception as ExceptionProto
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.Heading_pb2 import Heading as HeadingProto
from streamlit.proto.Json_pb2 import Json as JsonProto
from streamlit.proto.Markdown_pb2 import Markdown as MarkdownProto
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.proto.MultiSelect_pb2 import MultiSelect as MultiSelectProto
from streamlit.proto.NumberInput_pb2 import NumberInput as NumberInputProto
from streamlit.proto.Radio_pb2 import Radio as RadioProto
from streamlit.proto.Selectbox_pb2 import Selectbox as SelectboxProto
from streamlit.proto.Slider_pb2 import Slider as SliderProto
from streamlit.proto.Text_pb2 import Text as TextProto
from streamlit.proto.TextArea_pb2 import TextArea as TextAreaProto
from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto
from streamlit.proto.TimeInput_pb2 import TimeInput as TimeInputProto
from streamlit.proto.Toast_pb2 import Toast as ToastProto
from streamlit.proto.WidgetStates_pb2 import WidgetState, WidgetStates
from streamlit.runtime.state.common import TESTING_KEY, user_key_from_widget_id
from streamlit.runtime.state.safe_session_state import SafeSessionState
@dataclass(repr=False)
class TimeInput(Widget):
    """A representation of ``st.time_input``."""
    _value: TimeValue | None | InitialValue
    proto: TimeInputProto = field(repr=False)
    label: str
    step: int
    help: str
    form_id: str

    def __init__(self, proto: TimeInputProto, root: ElementTree):
        super().__init__(proto, root)
        self._value = InitialValue()
        self.type = 'time_input'

    def set_value(self, v: TimeValue | None) -> TimeInput:
        """Set the value of the widget."""
        self._value = v
        return self

    @property
    def _widget_state(self) -> WidgetState:
        ws = WidgetState()
        ws.id = self.id
        serde = TimeInputSerde(None)
        serialized_value = serde.serialize(self.value)
        if serialized_value is not None:
            ws.string_value = serialized_value
        return ws

    @property
    def value(self) -> time | None:
        """The current value of the widget. (time)"""
        if not isinstance(self._value, InitialValue):
            v = self._value
            v = v.time() if isinstance(v, datetime) else v
            return v
        else:
            state = self.root.session_state
            assert state
            return state[self.id]

    def increment(self) -> TimeInput:
        """Select the next available time."""
        if self.value is None:
            return self
        dt = datetime.combine(date.today(), self.value) + timedelta(seconds=self.step)
        return self.set_value(dt.time())

    def decrement(self) -> TimeInput:
        """Select the previous available time."""
        if self.value is None:
            return self
        dt = datetime.combine(date.today(), self.value) - timedelta(seconds=self.step)
        return self.set_value(dt.time())