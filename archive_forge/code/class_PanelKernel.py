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
class PanelKernel(Kernel):
    implementation = 'panel'
    implementation_version = __version__
    banner = 'banner'
    shell_stream = Any(ShellStream(), allow_none=True)

    def __init__(self, key=None, document=None):
        super().__init__()
        self.session = PanelSessionWebsocket(document=document, parent=self, key=key)
        self.stream = self.iopub_socket = WebsocketStream(self.session)
        self.io_loop = IOLoop.current()
        self.iopub_socket.channel = 'iopub'
        self.session.stream = self.iopub_socket
        self.comm_manager = CommManager(parent=self, kernel=self)
        self.shell = None
        self.session.auth = None
        self.log = logging.getLogger('fake')
        comm_msg_types = ['comm_open', 'comm_msg', 'comm_close']
        for msg_type in comm_msg_types:
            handler = getattr(self.comm_manager, msg_type)
            self.shell_handlers[msg_type] = self._wrap_handler(msg_type, handler)

    async def _flush_control_queue(self):
        pass

    def register_widget(self, widget):
        comm = widget.comm
        comm.kernel = self
        self.comm_manager.register_comm(comm)

    def _wrap_handler(self, msg_type, handler):
        doc = self.session._document

        def wrapper(*args, **kwargs):
            if msg_type == 'comm_open':
                return
            with set_curdoc(doc):
                state.execute(partial(handler, *args, **kwargs), schedule=True)
        return wrapper