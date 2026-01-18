import os
import re
import sys
import typing as t
from pathlib import Path
import zmq
from IPython.core.getipython import get_ipython
from IPython.core.inputtransformer2 import leading_empty_lines
from tornado.locks import Event
from tornado.queues import Queue
from zmq.utils import jsonapi
from .compiler import get_file_name, get_tmp_directory, get_tmp_hash_seed
class DebugpyClient:
    """A client for debugpy."""

    def __init__(self, log, debugpy_stream, event_callback):
        """Initialize the client."""
        self.log = log
        self.debugpy_stream = debugpy_stream
        self.event_callback = event_callback
        self.message_queue = DebugpyMessageQueue(self._forward_event, self.log)
        self.debugpy_host = '127.0.0.1'
        self.debugpy_port = -1
        self.routing_id = None
        self.wait_for_attach = True
        self.init_event = Event()
        self.init_event_seq = -1

    def _get_endpoint(self):
        host, port = self.get_host_port()
        return 'tcp://' + host + ':' + str(port)

    def _forward_event(self, msg):
        if msg['event'] == 'initialized':
            self.init_event.set()
            self.init_event_seq = msg['seq']
        self.event_callback(msg)

    def _send_request(self, msg):
        if self.routing_id is None:
            self.routing_id = self.debugpy_stream.socket.getsockopt(ROUTING_ID)
        content = jsonapi.dumps(msg, default=json_default, ensure_ascii=False, allow_nan=False)
        content_length = str(len(content))
        buf = (DebugpyMessageQueue.HEADER + content_length + DebugpyMessageQueue.SEPARATOR).encode('ascii')
        buf += content
        self.log.debug('DEBUGPYCLIENT:')
        self.log.debug(self.routing_id)
        self.log.debug(buf)
        self.debugpy_stream.send_multipart((self.routing_id, buf))

    async def _wait_for_response(self):
        return await self.message_queue.get_message()

    async def _handle_init_sequence(self):
        await self.init_event.wait()
        configurationDone = {'type': 'request', 'seq': int(self.init_event_seq) + 1, 'command': 'configurationDone'}
        self._send_request(configurationDone)
        await self._wait_for_response()
        return await self._wait_for_response()

    def get_host_port(self):
        """Get the host debugpy port."""
        if self.debugpy_port == -1:
            socket = self.debugpy_stream.socket
            socket.bind_to_random_port('tcp://' + self.debugpy_host)
            self.endpoint = socket.getsockopt(zmq.LAST_ENDPOINT).decode('utf-8')
            socket.unbind(self.endpoint)
            index = self.endpoint.rfind(':')
            self.debugpy_port = self.endpoint[index + 1:]
        return (self.debugpy_host, self.debugpy_port)

    def connect_tcp_socket(self):
        """Connect to the tcp socket."""
        self.debugpy_stream.socket.connect(self._get_endpoint())
        self.routing_id = self.debugpy_stream.socket.getsockopt(ROUTING_ID)

    def disconnect_tcp_socket(self):
        """Disconnect from the tcp socket."""
        self.debugpy_stream.socket.disconnect(self._get_endpoint())
        self.routing_id = None
        self.init_event = Event()
        self.init_event_seq = -1
        self.wait_for_attach = True

    def receive_dap_frame(self, frame):
        """Receive a dap frame."""
        self.message_queue.put_tcp_frame(frame)

    async def send_dap_request(self, msg):
        """Send a dap request."""
        self._send_request(msg)
        if self.wait_for_attach and msg['command'] == 'attach':
            rep = await self._handle_init_sequence()
            self.wait_for_attach = False
            return rep
        rep = await self._wait_for_response()
        self.log.debug('DEBUGPYCLIENT - returning:')
        self.log.debug(rep)
        return rep