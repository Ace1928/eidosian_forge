import logging
import os
from functools import partial
import ipykernel
import jupyter_client.session as session
import param
from bokeh.document.events import MessageSentEvent
from bokeh.document.json import Literal, MessageSent, TypedDict
from bokeh.util.serialization import make_id
from ipykernel.comm import Comm, CommManager
from ipykernel.kernelbase import Kernel
from ipywidgets import Widget
from ipywidgets._version import __protocol_version__
from ipywidgets.widgets.widget import _remove_buffers
from ipywidgets_bokeh.kernel import (
from ipywidgets_bokeh.widget import IPyWidget
from tornado.ioloop import IOLoop
from traitlets import Any
from ..config import __version__
from ..util import classproperty
from .state import set_curdoc, state
def _on_widget_constructed(widget, doc=None):
    doc = doc or state.curdoc
    if not doc or getattr(widget, '_document', None) not in (doc, None):
        return
    widget._document = doc
    kernel = _get_kernel(doc=doc)
    if widget.comm and widget.comm.target_name != 'panel-temp-comm' and (not (comm and isinstance(widget.comm, comm.DummyComm)) and isinstance(widget.comm.kernel, PanelKernel)):
        return
    wstate, buffer_paths, buffers = _remove_buffers(widget.get_state())
    args = {'target_name': 'jupyter.widget', 'data': {'state': wstate, 'buffer_paths': buffer_paths}, 'buffers': buffers, 'metadata': {'version': __protocol_version__}, 'kernel': kernel}
    if widget._model_id is not None:
        args['comm_id'] = widget._model_id
    try:
        widget.comm = Comm(**args)
    except Exception as e:
        if 'PANEL_IPYWIDGET' not in os.environ:
            raise e
    kernel.register_widget(widget)