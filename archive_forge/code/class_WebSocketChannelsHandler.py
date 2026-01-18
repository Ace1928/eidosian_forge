from __future__ import annotations
import asyncio
import logging
import mimetypes
import os
import random
import warnings
from typing import Any, Optional, cast
from jupyter_client.session import Session
from tornado import web
from tornado.concurrent import Future
from tornado.escape import json_decode, url_escape, utf8
from tornado.httpclient import HTTPRequest
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.websocket import WebSocketHandler, websocket_connect
from traitlets.config.configurable import LoggingConfigurable
from ..base.handlers import APIHandler, JupyterHandler
from ..utils import url_path_join
from .gateway_client import GatewayClient
from ..services.kernels.handlers import _kernel_id_regex
from ..services.kernelspecs.handlers import kernel_name_regex
class WebSocketChannelsHandler(WebSocketHandler, JupyterHandler):
    """Gateway web socket channels handler."""
    session = None
    gateway = None
    kernel_id = None
    ping_callback = None

    def check_origin(self, origin=None):
        """Check origin for the socket."""
        return JupyterHandler.check_origin(self, origin)

    def set_default_headers(self):
        """Undo the set_default_headers in JupyterHandler which doesn't make sense for websockets"""

    def get_compression_options(self):
        """Get the compression options for the socket."""
        return {}

    def authenticate(self):
        """Run before finishing the GET request

        Extend this method to add logic that should fire before
        the websocket finishes completing.
        """
        if self.current_user is None:
            self.log.warning("Couldn't authenticate WebSocket connection")
            raise web.HTTPError(403)
        if self.get_argument('session_id', None):
            assert self.session is not None
            self.session.session = self.get_argument('session_id')
        else:
            self.log.warning('No session ID specified')

    def initialize(self):
        """Initialize the socket."""
        self.log.debug('Initializing websocket connection %s', self.request.path)
        self.session = Session(config=self.config)
        self.gateway = GatewayWebSocketClient(gateway_url=GatewayClient.instance().url)

    async def get(self, kernel_id, *args, **kwargs):
        """Get the socket."""
        self.authenticate()
        self.kernel_id = kernel_id
        kwargs['kernel_id'] = kernel_id
        await super().get(*args, **kwargs)

    def send_ping(self):
        """Send a ping to the socket."""
        if self.ws_connection is None and self.ping_callback is not None:
            self.ping_callback.stop()
            return
        self.ping(b'')

    def open(self, kernel_id, *args, **kwargs):
        """Handle web socket connection open to notebook server and delegate to gateway web socket handler"""
        self.ping_callback = PeriodicCallback(self.send_ping, GATEWAY_WS_PING_INTERVAL_SECS * 1000)
        self.ping_callback.start()
        assert self.gateway is not None
        self.gateway.on_open(kernel_id=kernel_id, message_callback=self.write_message, compression_options=self.get_compression_options())

    def on_message(self, message):
        """Forward message to gateway web socket handler."""
        assert self.gateway is not None
        self.gateway.on_message(message)

    def write_message(self, message, binary=False):
        """Send message back to notebook client.  This is called via callback from self.gateway._read_messages."""
        if self.ws_connection:
            if isinstance(message, bytes):
                binary = True
            super().write_message(message, binary=binary)
        elif self.log.isEnabledFor(logging.DEBUG):
            msg_summary = WebSocketChannelsHandler._get_message_summary(json_decode(utf8(message)))
            self.log.debug(f'Notebook client closed websocket connection - message dropped: {msg_summary}')

    def on_close(self):
        """Handle a closing socket."""
        self.log.debug('Closing websocket connection %s', self.request.path)
        assert self.gateway is not None
        self.gateway.on_close()
        super().on_close()

    @staticmethod
    def _get_message_summary(message):
        """Get a summary of a message."""
        summary = []
        message_type = message['msg_type']
        summary.append(f'type: {message_type}')
        if message_type == 'status':
            summary.append(', state: {}'.format(message['content']['execution_state']))
        elif message_type == 'error':
            summary.append(', {}:{}:{}'.format(message['content']['ename'], message['content']['evalue'], message['content']['traceback']))
        else:
            summary.append(', ...')
        return ''.join(summary)