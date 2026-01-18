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
class Multiselect(Widget, Generic[T]):
    """A representation of ``st.multiselect``."""
    _value: list[T] | None
    proto: MultiSelectProto = field(repr=False)
    label: str
    options: list[str]
    max_selections: int
    help: str
    form_id: str

    def __init__(self, proto: MultiSelectProto, root: ElementTree):
        super().__init__(proto, root)
        self.type = 'multiselect'
        self.options = list(proto.options)

    @property
    def _widget_state(self) -> WidgetState:
        """Protobuf message representing the state of the widget, including
        any interactions that have happened.
        Should be the same as the frontend would produce for those interactions.
        """
        ws = WidgetState()
        ws.id = self.id
        ws.int_array_value.data[:] = self.indices
        return ws

    @property
    def value(self) -> list[T]:
        """The currently selected values from the options. (list)"""
        if self._value is not None:
            return self._value
        else:
            state = self.root.session_state
            assert state
            return cast(List[T], state[self.id])

    @property
    def indices(self) -> Sequence[int]:
        """The indices of the currently selected values from the options. (list)"""
        return [self.options.index(self.format_func(v)) for v in self.value]

    @property
    def format_func(self) -> Callable[[Any], Any]:
        """The widget's formatting function for displaying options. (callable)"""
        ss = self.root.session_state
        return cast(Callable[[Any], Any], ss[TESTING_KEY][self.id])

    def set_value(self, v: list[T]) -> Multiselect[T]:
        """Set the value of the multiselect widget. (list)"""
        self._value = v
        return self

    def select(self, v: T) -> Multiselect[T]:
        """
        Add a selection to the widget. Do nothing if the value is already selected.        If testing a multiselect widget with repeated options, use ``set_value``        instead.
        """
        current = self.value
        if v in current:
            return self
        else:
            new = current.copy()
            new.append(v)
            self.set_value(new)
            return self

    def unselect(self, v: T) -> Multiselect[T]:
        """
        Remove a selection from the widget. Do nothing if the value is not        already selected. If a value is selected multiple times, the first        instance is removed.
        """
        current = self.value
        if v not in current:
            return self
        else:
            new = current.copy()
            while v in new:
                new.remove(v)
            self.set_value(new)
            return self