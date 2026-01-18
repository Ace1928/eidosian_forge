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
def _prep_data_for_add_rows(data: Data, delta_type: str, add_rows_metadata: AddRowsMetadata) -> tuple[Data, AddRowsMetadata]:
    out_data: Data
    if delta_type in ARROW_DELTA_TYPES_THAT_MELT_DATAFRAMES:
        import pandas as pd
        df = cast(pd.DataFrame, type_util.convert_anything_to_df(data))
        if isinstance(df.index, pd.RangeIndex):
            old_step = _get_pandas_index_attr(df, 'step')
            df = df.reset_index(drop=True)
            old_stop = _get_pandas_index_attr(df, 'stop')
            if old_step is None or old_stop is None:
                raise StreamlitAPIException("'RangeIndex' object has no attribute 'step'")
            start = add_rows_metadata.last_index + old_step
            stop = add_rows_metadata.last_index + old_step + old_stop
            df.index = pd.RangeIndex(start=start, stop=stop, step=old_step)
            add_rows_metadata.last_index = stop - 1
        out_data, *_ = prep_data(df, **add_rows_metadata.columns)
    else:
        out_data = type_util.convert_anything_to_df(data, allow_styler=True)
    return (out_data, add_rows_metadata)