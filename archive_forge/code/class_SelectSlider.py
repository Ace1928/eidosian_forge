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
class SelectSlider(Widget, Generic[T]):
    """A representation of ``st.select_slider``."""
    _value: T | Sequence[T] | None
    proto: SliderProto = field(repr=False)
    label: str
    data_type: SliderProto.DataType.ValueType
    options: list[str]
    help: str
    form_id: str

    def __init__(self, proto: SliderProto, root: ElementTree):
        super().__init__(proto, root)
        self.type = 'select_slider'
        self.options = list(proto.options)

    def set_value(self, v: T | Sequence[T]) -> SelectSlider[T]:
        """Set the (single) selection by value."""
        self._value = v
        return self

    @property
    def _widget_state(self) -> WidgetState:
        serde = SelectSliderSerde(self.options, [], False)
        try:
            v = serde.serialize(self.format_func(self.value))
        except (ValueError, TypeError):
            try:
                v = serde.serialize([self.format_func(val) for val in self.value])
            except:
                raise ValueError(f'Could not find index for {self.value}')
        ws = WidgetState()
        ws.id = self.id
        ws.double_array_value.data[:] = v
        return ws

    @property
    def value(self) -> T | Sequence[T]:
        """The currently selected value or range. (Any or Sequence of Any)"""
        if self._value is not None:
            return self._value
        else:
            state = self.root.session_state
            assert state
            return state[self.id]

    @property
    def format_func(self) -> Callable[[Any], Any]:
        """The widget's formatting function for displaying options. (callable)"""
        ss = self.root.session_state
        return cast(Callable[[Any], Any], ss[TESTING_KEY][self.id])

    def set_range(self, lower: T, upper: T) -> SelectSlider[T]:
        """Set the ranged selection by values."""
        return self.set_value([lower, upper])