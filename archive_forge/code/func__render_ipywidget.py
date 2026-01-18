import base64
import os
from contextlib import contextmanager
from functools import partial
from io import BytesIO, StringIO
import panel as pn
import param
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.io import curdoc
from bokeh.resources import CDN, INLINE
from packaging.version import Version
from panel import config
from panel.io.notebook import ipywidget, load_notebook, render_mimebundle, render_model
from panel.io.state import state
from panel.models.comm_manager import CommManager as PnCommManager
from panel.pane import HoloViews as HoloViewsPane
from panel.viewable import Viewable
from panel.widgets.player import PlayerBase
from pyviz_comms import CommManager
from param.parameterized import bothmethod
from ..core import AdjointLayout, DynamicMap, HoloMap, Layout
from ..core.data import disable_pipeline
from ..core.io import Exporter
from ..core.options import Compositor, SkipRendering, Store, StoreOptions
from ..core.util import unbound_dimensions
from ..streams import Stream
from . import Plot
from .util import collate, displayable, initialize_dynamic
def _render_ipywidget(self, plot):
    widget = ipywidget(plot, combine_events=True)
    if hasattr(widget, '_repr_mimebundle_'):
        return (widget._repr_mimebundle_(), {})
    plaintext = repr(widget)
    if len(plaintext) > 110:
        plaintext = plaintext[:110] + '…'
    data = {'text/plain': plaintext}
    if widget._view_name is not None:
        data['application/vnd.jupyter.widget-view+json'] = {'version_major': 2, 'version_minor': 0, 'model_id': widget._model_id}
    if config.comms == 'vscode':
        from IPython.display import display
        display(data, raw=True)
        return ({'text/html': '<div style="display: none"></div>'}, {})
    return (data, {})