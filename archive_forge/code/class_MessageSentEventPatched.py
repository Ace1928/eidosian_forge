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
class MessageSentEventPatched(MessageSentEvent):
    """
    Patches MessageSentEvent with fix that ensures MessageSent event
    does not define msg_data (which is an assumption in BokehJS
    Document.apply_json_patch.)
    """

    def generate(self, references, buffers):
        if not isinstance(self.msg_data, bytes):
            msg = MessageSent(kind=self.kind, msg_type=self.msg_type, msg_data=self.msg_data)
        else:
            msg = MessageSentBuffers(kind=self.kind, msg_type=self.msg_type)
            assert buffers is not None
            buffer_id = make_id()
            buf = (dict(id=buffer_id), self.msg_data)
            buffers.append(buf)
        return msg