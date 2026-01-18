from the server to the kernel.
from __future__ import annotations
import asyncio
import calendar
import datetime as dt
import inspect
import json
import logging
import os
import pathlib
import textwrap
import time
from queue import Empty
from typing import Any, Awaitable
from urllib.parse import urljoin
import tornado
from bokeh.embed.bundle import extension_dirs
from bokeh.protocol import Protocol
from bokeh.protocol.exceptions import ProtocolError
from bokeh.protocol.receiver import Receiver
from bokeh.server.tornado import DEFAULT_KEEP_ALIVE_MS
from bokeh.server.views.multi_root_static_handler import MultiRootStaticHandler
from bokeh.server.views.static_handler import StaticHandler
from bokeh.server.views.ws import WSHandler
from bokeh.util.token import (
from jupyter_server.base.handlers import JupyterHandler
from tornado.ioloop import PeriodicCallback
from tornado.web import StaticFileHandler
from ..config import config
from .resources import DIST_DIR, ERROR_TEMPLATE, _env
from .server import COMPONENT_PATH, ComponentResourceHandler
from .state import state
import os
import pathlib
import sys
from panel.io.jupyter_executor import PanelExecutor
class PanelWSProxy(WSHandler, JupyterHandler):
    """
    The PanelWSProxy serves as a proxy between the frontend and the
    Jupyter kernel that is running the Panel application. It send and
    receives Bokeh protocol messages via a Jupyter Comm.
    """
    _tasks = set()

    def __init__(self, tornado_app, *args, **kw) -> None:
        kw['application_context'] = None
        super().__init__(tornado_app, *args, **kw)

    def initialize(self, *args, **kwargs):
        self._ping_count = 0
        self._ping_job = PeriodicCallback(self._keep_alive, DEFAULT_KEEP_ALIVE_MS)

    def _keep_alive(self):
        self.ping(str(self._ping_count).encode('utf-8'))
        self._ping_count += 1

    async def prepare(self):
        pass

    def get_current_user(self):
        return 'default_user'

    def check_origin(self, origin: str) -> bool:
        return True

    @tornado.web.authenticated
    async def open(self, path, *args, **kwargs) -> None:
        """ Initialize a connection to a client.

        Returns:
            None

        """
        token = self._token
        if self.selected_subprotocol != 'bokeh':
            self.close()
            raise ProtocolError("Subprotocol header is not 'bokeh'")
        elif token is None:
            self.close()
            raise ProtocolError('No token received in subprotocol header')
        now = calendar.timegm(dt.datetime.utcnow().utctimetuple())
        payload = get_token_payload(token)
        if 'session_expiry' not in payload:
            self.close()
            raise ProtocolError('Session expiry has not been provided')
        elif now >= payload['session_expiry']:
            self.close()
            raise ProtocolError('Token is expired.')
        try:
            protocol = Protocol()
            self.receiver = Receiver(protocol)
        except ProtocolError as e:
            logger.error('Could not create new server session, reason: %s', e)
            self.close()
            raise e
        self.session_id = get_session_id(token)
        if self.session_id not in state._kernels:
            self.close()
        kernel_info = state._kernels[self.session_id]
        self.kernel, self.comm_id, self.kernel_id, _ = kernel_info
        state._kernels[self.session_id] = kernel_info[:-1] + (True,)
        msg = protocol.create('ACK')
        await self.send_message(msg)
        self._ping_job.start()
        task = asyncio.create_task(self._check_for_message())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _check_for_message(self):
        while True:
            if self.kernel is None:
                break
            try:
                msg = await ensure_async(self.kernel.iopub_channel.get_msg(timeout=None))
            except Empty as e:
                if not await ensure_async(self.kernel.is_alive()):
                    raise RuntimeError('Kernel died before expected shutdown of Panel app.') from e
                continue
            msg_type = msg['header']['msg_type']
            if msg_type == 'stream' and msg['content']['name'] == 'stderr':
                logger.error(msg['content']['text'])
                continue
            elif not (msg_type == 'comm_msg' and msg['content']['comm_id'] == self.comm_id):
                continue
            content, metadata = (msg['content'], msg['metadata'])
            status = metadata.get('status')
            if status == 'protocol_error':
                return self._protocol_error(content['data'])
            elif status == 'internal_error':
                return self._internal_error(content['data'])
            binary = metadata.get('binary')
            if binary:
                fragment = msg['buffers'][0].tobytes()
            else:
                fragment = content['data']
                if isinstance(fragment, dict):
                    fragment = json.dumps(fragment)
            message = await self._receive(fragment)
            if message:
                await self.send_message(message)

    async def on_message(self, fragment: str | bytes) -> None:
        content = dict(data=fragment, comm_id=self.comm_id, target_name=self.session_id)
        msg = self.kernel.session.msg('comm_msg', content)
        self.kernel.shell_channel.send(msg)

    def on_close(self) -> None:
        """
        Clean up when the connection is closed.
        """
        logger.info('WebSocket connection closed: code=%s, reason=%r', self.close_code, self.close_reason)
        if self.session_id in state._kernels:
            del state._kernels[self.session_id]
        self._ping_job.stop()
        self._shutdown_futures = [asyncio.ensure_future(self.kernel.shutdown(reply=True)), asyncio.ensure_future(self.kernel_manager.shutdown_kernel(self.kernel_id, now=True))]
        self.kernel = None