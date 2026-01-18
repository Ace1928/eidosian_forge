from __future__ import annotations
import asyncio
import datetime
import json
import os
from logging import Logger
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Any, Optional, cast
import websocket
from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.clientabc import KernelClientABC
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.managerabc import KernelManagerABC
from jupyter_core.utils import ensure_async
from tornado import web
from tornado.escape import json_decode, json_encode, url_escape, utf8
from traitlets import DottedObjectName, Instance, Type, default
from .._tz import UTC, utcnow
from ..services.kernels.kernelmanager import (
from ..services.sessions.sessionmanager import SessionManager
from ..utils import url_path_join
from .gateway_client import GatewayClient, gateway_request
class GatewayKernelManager(ServerKernelManager):
    """Manages a single kernel remotely via a Gateway Server."""
    kernel_id: Optional[str] = None
    kernel = None

    @default('cache_ports')
    def _default_cache_ports(self):
        return False

    def __init__(self, **kwargs):
        """Initialize the gateway kernel manager."""
        super().__init__(**kwargs)
        self.kernels_url = url_path_join(GatewayClient.instance().url or '', GatewayClient.instance().kernels_endpoint)
        self.kernel_url: str
        self.kernel = self.kernel_id = None
        self.execution_state = 'starting'
        self.last_activity = utcnow()

    @property
    def has_kernel(self):
        """Has a kernel been started that we are managing."""
        return self.kernel is not None
    client_class = DottedObjectName('jupyter_server.gateway.managers.GatewayKernelClient')
    client_factory = Type(klass='jupyter_server.gateway.managers.GatewayKernelClient')

    def client(self, **kwargs):
        """Create a client configured to connect to our kernel"""
        kw: dict[str, Any] = {}
        kw.update(self.get_connection_info(session=True))
        kw.update({'connection_file': self.connection_file, 'parent': self})
        kw['kernel_id'] = self.kernel_id
        kw.update(kwargs)
        return self.client_factory(**kw)

    async def refresh_model(self, model=None):
        """Refresh the kernel model.

        Parameters
        ----------
        model : dict
            The model from which to refresh the kernel.  If None, the kernel
            model is fetched from the Gateway server.
        """
        if model is None:
            self.log.debug('Request kernel at: %s' % self.kernel_url)
            try:
                response = await gateway_request(self.kernel_url, method='GET')
            except web.HTTPError as error:
                if error.status_code == 404:
                    self.log.warning('Kernel not found at: %s' % self.kernel_url)
                    model = None
                else:
                    raise
            else:
                model = json_decode(response.body)
            self.log.debug('Kernel retrieved: %s' % model)
        if model:
            self.last_activity = datetime.datetime.strptime(model['last_activity'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=UTC)
            self.execution_state = model['execution_state']
            if isinstance(self.parent, AsyncMappingKernelManager):
                self.parent._kernel_connections[self.kernel_id] = int(model['connections'])
        self.kernel = model
        return model

    @emit_kernel_action_event(success_msg='Kernel {kernel_id} was started.')
    async def start_kernel(self, **kwargs):
        """Starts a kernel via HTTP in an asynchronous manner.

        Parameters
        ----------
        `**kwargs` : optional
             keyword arguments that are passed down to build the kernel_cmd
             and launching the kernel (e.g. Popen kwargs).
        """
        kernel_id = kwargs.get('kernel_id')
        if kernel_id is None:
            kernel_name = kwargs.get('kernel_name', 'python3')
            self.log.debug('Request new kernel at: %s' % self.kernels_url)
            if os.environ.get('KERNEL_USERNAME') is None and GatewayClient.instance().http_user:
                os.environ['KERNEL_USERNAME'] = GatewayClient.instance().http_user or ''
            payload_envs = os.environ.copy()
            payload_envs.update(kwargs.get('env', {}))
            kernel_env = {k: v for k, v in payload_envs.items() if k.startswith('KERNEL_') or k in GatewayClient.instance().allowed_envs.split(',')}
            if kwargs.get('cwd') is not None and kernel_env.get('KERNEL_WORKING_DIR') is None:
                kernel_env['KERNEL_WORKING_DIR'] = kwargs['cwd']
            json_body = json_encode({'name': kernel_name, 'env': kernel_env})
            response = await gateway_request(self.kernels_url, method='POST', headers={'Content-Type': 'application/json'}, body=json_body)
            self.kernel = json_decode(response.body)
            self.kernel_id = self.kernel['id']
            self.kernel_url = url_path_join(self.kernels_url, url_escape(str(self.kernel_id)))
            self.log.info(f'GatewayKernelManager started kernel: {self.kernel_id}, args: {kwargs}')
        else:
            self.kernel_id = kernel_id
            self.kernel_url = url_path_join(self.kernels_url, url_escape(str(self.kernel_id)))
            self.kernel = await self.refresh_model()
            self.log.info(f'GatewayKernelManager using existing kernel: {self.kernel_id}')

    @emit_kernel_action_event(success_msg='Kernel {kernel_id} was shutdown.')
    async def shutdown_kernel(self, now=False, restart=False):
        """Attempts to stop the kernel process cleanly via HTTP."""
        if self.has_kernel:
            self.log.debug('Request shutdown kernel at: %s', self.kernel_url)
            try:
                response = await gateway_request(self.kernel_url, method='DELETE')
                self.log.debug('Shutdown kernel response: %d %s', response.code, response.reason)
            except web.HTTPError as error:
                if error.status_code == 404:
                    self.log.debug('Shutdown kernel response: kernel not found (ignored)')
                else:
                    raise

    @emit_kernel_action_event(success_msg='Kernel {kernel_id} was restarted.')
    async def restart_kernel(self, **kw):
        """Restarts a kernel via HTTP."""
        if self.has_kernel:
            assert self.kernel_url is not None
            kernel_url = self.kernel_url + '/restart'
            self.log.debug('Request restart kernel at: %s', kernel_url)
            response = await gateway_request(kernel_url, method='POST', headers={'Content-Type': 'application/json'}, body=json_encode({}))
            self.log.debug('Restart kernel response: %d %s', response.code, response.reason)

    @emit_kernel_action_event(success_msg='Kernel {kernel_id} was interrupted.')
    async def interrupt_kernel(self):
        """Interrupts the kernel via an HTTP request."""
        if self.has_kernel:
            assert self.kernel_url is not None
            kernel_url = self.kernel_url + '/interrupt'
            self.log.debug('Request interrupt kernel at: %s', kernel_url)
            response = await gateway_request(kernel_url, method='POST', headers={'Content-Type': 'application/json'}, body=json_encode({}))
            self.log.debug('Interrupt kernel response: %d %s', response.code, response.reason)

    async def is_alive(self):
        """Is the kernel process still running?"""
        if self.has_kernel:
            self.kernel = await self.refresh_model()
            self.log.debug(f'The kernel: {self.kernel} is alive.')
            return True
        else:
            self.log.debug(f'The kernel: {self.kernel} no longer exists.')
            return False

    def cleanup_resources(self, restart=False):
        """Clean up resources when the kernel is shut down"""