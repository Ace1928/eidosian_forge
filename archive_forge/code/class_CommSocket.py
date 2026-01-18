from base64 import b64encode
import io
import json
import pathlib
import uuid
from ipykernel.comm import Comm
from IPython.display import display, Javascript, HTML
from matplotlib import is_interactive
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import _Backend, CloseEvent, NavigationToolbar2
from .backend_webagg_core import (
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611
class CommSocket:
    """
    Manages the Comm connection between IPython and the browser (client).

    Comms are 2 way, with the CommSocket being able to publish a message
    via the send_json method, and handle a message with on_message. On the
    JS side figure.send_message and figure.ws.onmessage do the sending and
    receiving respectively.

    """

    def __init__(self, manager):
        self.supports_binary = None
        self.manager = manager
        self.uuid = str(uuid.uuid4())
        display(HTML('<div id=%r></div>' % self.uuid))
        try:
            self.comm = Comm('matplotlib', data={'id': self.uuid})
        except AttributeError as err:
            raise RuntimeError('Unable to create an IPython notebook Comm instance. Are you in the IPython notebook?') from err
        self.comm.on_msg(self.on_message)
        manager = self.manager
        self._ext_close = False

        def _on_close(close_message):
            self._ext_close = True
            manager.remove_comm(close_message['content']['comm_id'])
            manager.clearup_closed()
        self.comm.on_close(_on_close)

    def is_open(self):
        return not (self._ext_close or self.comm._closed)

    def on_close(self):
        if self.is_open():
            try:
                self.comm.close()
            except KeyError:
                pass

    def send_json(self, content):
        self.comm.send({'data': json.dumps(content)})

    def send_binary(self, blob):
        if self.supports_binary:
            self.comm.send({'blob': 'image/png'}, buffers=[blob])
        else:
            data = b64encode(blob).decode('ascii')
            data_uri = f'data:image/png;base64,{data}'
            self.comm.send({'data': data_uri})

    def on_message(self, message):
        message = json.loads(message['content']['data'])
        if message['type'] == 'closing':
            self.on_close()
            self.manager.clearup_closed()
        elif message['type'] == 'supports_binary':
            self.supports_binary = message['value']
        else:
            self.manager.handle_json(message)