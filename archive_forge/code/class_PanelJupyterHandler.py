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
class PanelJupyterHandler(PanelBaseHandler):
    """
    The PanelJupyterHandler expects to be given a path to a notebook,
    .py file or Lumen .yaml file. Based on the kernelspec in the
    notebook or the kernel query parameter it will then provision
    a Jupyter kernel to run the Panel application in.

    Once the kernel is launched it will instantiate a PanelExecutor
    inside the kernel and serve the HTML returned by it. If successful
    it will store the kernel and comm_id on `panel.state`.
    """

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.kernel_started = False

    async def _get_info(self, msg_id, timeout=KERNEL_TIMEOUT):
        deadline = time.monotonic() + timeout
        result, comm_id, extension_dirs = (None, None, None)
        while result is None or comm_id is None or extension_dirs is None:
            if time.monotonic() > deadline:
                raise TimeoutError('Timed out while waiting for kernel to open Comm channel to Panel application.')
            try:
                msg = await ensure_async(self.kernel.iopub_channel.get_msg(timeout=None))
            except Empty as e:
                if not await ensure_async(self.kernel.is_alive()):
                    raise RuntimeError('Kernel died before establishing Comm connection to Panel application.') from e
                continue
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            msg_type = msg['header']['msg_type']
            if msg_type == 'execute_result':
                data = msg['content']['data']
                if 'text/error' in data:
                    raise RuntimeError(data['text/error'])
                extension_dirs = data['application/bokeh-extensions']
                result = data['text/html']
            elif msg_type == 'comm_open' and msg['content']['target_name'] == self.session_id:
                comm_id = msg['content']['comm_id']
            elif msg_type == 'stream' and msg['content']['name'] == 'stderr':
                logger.error(msg['content']['text'])
            elif msg_type == 'error':
                logger.error(msg['content']['traceback'])
        return (result, comm_id, extension_dirs)

    @tornado.web.authenticated
    async def get(self, path=None):
        notebook_path = self.nb_path(path)
        if self.notebook_path and path:
            self.redirect_to_file(path)
            return
        cwd = os.path.dirname(notebook_path)
        root_url = url_path_join(self.base_url, 'panel-preview')
        self.set_header('Content-Type', 'text/html')
        self.set_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.set_header('Pragma', 'no-cache')
        self.set_header('Expires', '0')
        if self.request.arguments.get('kernel'):
            requested_kernel = self.request.arguments.pop('kernel')[0].decode('utf-8')
        elif notebook_path.suffix == '.ipynb':
            with open(notebook_path) as f:
                nb = json.load(f)
            requested_kernel = nb.get('metadata', {}).get('kernelspec', {}).get('name')
        else:
            requested_kernel = None
        if requested_kernel:
            available_kernels = list(self.kernel_manager.kernel_spec_manager.find_kernel_specs())
            if requested_kernel not in available_kernels:
                logger.error('Could not start server session, no such kernel %r.', requested_kernel)
                html = KERNEL_ERROR_TEMPLATE.render(base_url=f'{root_url}/', kernels=available_kernels, error_type='Kernel Error', error=f"No such kernel '{requested_kernel}'", title='Panel: Kernel not found')
                self.finish(html)
                return
        kernel_env = {**os.environ}
        kernel_id = await ensure_async(self.kernel_manager.start_kernel(kernel_name=requested_kernel, path=cwd, env=kernel_env))
        kernel_future = self.kernel_manager.get_kernel(kernel_id)
        km = await ensure_async(kernel_future)
        self.kernel = km.client()
        await ensure_async(self.kernel.start_channels())
        await ensure_async(self.kernel.wait_for_ready(timeout=None))
        self.session_id = generate_session_id()
        args = {k: [v.decode('utf-8') for v in vs] for k, vs in self.request.arguments.items()}
        payload = {'arguments': args, 'headers': dict(self.request.headers.items()), 'cookies': dict(self.request.cookies.items())}
        token = generate_jwt_token(self.session_id, extra_payload=payload)
        source = generate_executor(notebook_path, token, root_url)
        msg_id = self.kernel.execute(source)
        try:
            html, comm_id, ext_dirs = await self._get_info(msg_id)
        except (TimeoutError, RuntimeError) as e:
            await self.kernel_manager.shutdown_kernel(kernel_id, now=True)
            html = ERROR_TEMPLATE.render(npm_cdn=config.npm_cdn, base_url=f'{root_url}/', error_type='Kernel Error', error='Failed to start application', error_msg=str(e), title='Panel: Kernel Error')
            self.finish(html)
        else:
            extension_dirs.update(ext_dirs)
            state._kernels[self.session_id] = (self.kernel, comm_id, kernel_id, False)
            loop = tornado.ioloop.IOLoop.current()
            loop.call_later(CONNECTION_TIMEOUT, self._check_connected)
            self.finish(html)

    async def _check_connected(self):
        if self.session_id not in state._kernels:
            return
        _, _, kernel_id, connected = state._kernels[self.session_id]
        if not connected:
            await self.kernel_manager.shutdown_kernel(kernel_id, now=True)