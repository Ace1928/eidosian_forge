from __future__ import annotations
import sys
from contextvars import ContextVar
from copy import deepcopy
from typing import (
from streamlit import (
from streamlit.cursor import Cursor
from streamlit.elements.alert import AlertMixin
from streamlit.elements.altair_utils import AddRowsMetadata
from streamlit.elements.arrow import ArrowMixin
from streamlit.elements.arrow_altair import ArrowAltairMixin, prep_data
from streamlit.elements.arrow_vega_lite import ArrowVegaLiteMixin
from streamlit.elements.balloons import BalloonsMixin
from streamlit.elements.bokeh_chart import BokehMixin
from streamlit.elements.code import CodeMixin
from streamlit.elements.deck_gl_json_chart import PydeckMixin
from streamlit.elements.doc_string import HelpMixin
from streamlit.elements.empty import EmptyMixin
from streamlit.elements.exception import ExceptionMixin
from streamlit.elements.form import FormData, FormMixin, current_form_id
from streamlit.elements.graphviz_chart import GraphvizMixin
from streamlit.elements.heading import HeadingMixin
from streamlit.elements.html import HtmlMixin
from streamlit.elements.iframe import IframeMixin
from streamlit.elements.image import ImageMixin
from streamlit.elements.json import JsonMixin
from streamlit.elements.layouts import LayoutsMixin
from streamlit.elements.map import MapMixin
from streamlit.elements.markdown import MarkdownMixin
from streamlit.elements.media import MediaMixin
from streamlit.elements.metric import MetricMixin
from streamlit.elements.plotly_chart import PlotlyMixin
from streamlit.elements.progress import ProgressMixin
from streamlit.elements.pyplot import PyplotMixin
from streamlit.elements.snow import SnowMixin
from streamlit.elements.text import TextMixin
from streamlit.elements.toast import ToastMixin
from streamlit.elements.widgets.button import ButtonMixin
from streamlit.elements.widgets.camera_input import CameraInputMixin
from streamlit.elements.widgets.chat import ChatMixin
from streamlit.elements.widgets.checkbox import CheckboxMixin
from streamlit.elements.widgets.color_picker import ColorPickerMixin
from streamlit.elements.widgets.data_editor import DataEditorMixin
from streamlit.elements.widgets.file_uploader import FileUploaderMixin
from streamlit.elements.widgets.multiselect import MultiSelectMixin
from streamlit.elements.widgets.number_input import NumberInputMixin
from streamlit.elements.widgets.radio import RadioMixin
from streamlit.elements.widgets.select_slider import SelectSliderMixin
from streamlit.elements.widgets.selectbox import SelectboxMixin
from streamlit.elements.widgets.slider import SliderMixin
from streamlit.elements.widgets.text_widgets import TextWidgetsMixin
from streamlit.elements.widgets.time_widgets import TimeWidgetsMixin
from streamlit.elements.write import WriteMixin
from streamlit.errors import NoSessionContext, StreamlitAPIException
from streamlit.proto import Block_pb2, ForwardMsg_pb2
from streamlit.proto.RootContainer_pb2 import RootContainer
from streamlit.runtime import caching, legacy_caching
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import NoValue
def _enqueue_message(msg: ForwardMsg_pb2.ForwardMsg) -> None:
    """Enqueues a ForwardMsg proto to send to the app."""
    ctx = get_script_run_ctx()
    if ctx is None:
        raise NoSessionContext()
    if ctx.current_fragment_id and msg.WhichOneof('type') == 'delta':
        msg.delta.fragment_id = ctx.current_fragment_id
    ctx.enqueue(msg)