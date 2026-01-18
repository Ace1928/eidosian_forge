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
class MessageSentBuffers(TypedDict):
    kind: Literal['MessageSent']
    msg_type: str