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
class GatewayKernelClient(AsyncKernelClient):
    """Communicates with a single kernel indirectly via a websocket to a gateway server.

    There are five channels associated with each kernel:

    * shell: for request/reply calls to the kernel.
    * iopub: for the kernel to publish results to frontends.
    * hb: for monitoring the kernel's heartbeat.
    * stdin: for frontends to reply to raw_input calls in the kernel.
    * control: for kernel management calls to the kernel.

    The messages that can be sent on these channels are exposed as methods of the
    client (KernelClient.execute, complete, history, etc.). These methods only
    send the message, they don't wait for a reply. To get results, use e.g.
    :meth:`get_shell_msg` to fetch messages from the shell channel.
    """
    allow_stdin = False
    _channels_stopped: bool
    _channel_queues: Optional[dict[str, ChannelQueue]]
    _control_channel: Optional[ChannelQueue]
    _hb_channel: Optional[ChannelQueue]
    _stdin_channel: Optional[ChannelQueue]
    _iopub_channel: Optional[ChannelQueue]
    _shell_channel: Optional[ChannelQueue]

    def __init__(self, kernel_id, **kwargs):
        """Initialize a gateway kernel client."""
        super().__init__(**kwargs)
        self.kernel_id = kernel_id
        self.channel_socket: Optional[websocket.WebSocket] = None
        self.response_router: Optional[Thread] = None
        self._channels_stopped = False
        self._channel_queues = {}

    async def start_channels(self, shell=True, iopub=True, stdin=True, hb=True, control=True):
        """Starts the channels for this kernel.

        For this class, we establish a websocket connection to the destination
        and set up the channel-based queues on which applicable messages will
        be posted.
        """
        ws_url = url_path_join(GatewayClient.instance().ws_url or '', GatewayClient.instance().kernels_endpoint, url_escape(self.kernel_id), 'channels')
        ssl_options = {'ca_certs': GatewayClient.instance().ca_certs, 'certfile': GatewayClient.instance().client_cert, 'keyfile': GatewayClient.instance().client_key}
        self.channel_socket = websocket.create_connection(ws_url, timeout=GatewayClient.instance().KERNEL_LAUNCH_TIMEOUT, enable_multithread=True, sslopt=ssl_options)
        await ensure_async(super().start_channels(shell=shell, iopub=iopub, stdin=stdin, hb=hb, control=control))
        self.response_router = Thread(target=self._route_responses)
        self.response_router.start()

    def stop_channels(self):
        """Stops all the running channels for this kernel.

        For this class, we close the websocket connection and destroy the
        channel-based queues.
        """
        super().stop_channels()
        self._channels_stopped = True
        self.log.debug('Closing websocket connection')
        assert self.channel_socket is not None
        self.channel_socket.close()
        assert self.response_router is not None
        self.response_router.join()
        if self._channel_queues:
            self._channel_queues.clear()
            self._channel_queues = None

    @property
    def shell_channel(self):
        """Get the shell channel object for this kernel."""
        if self._shell_channel is None:
            self.log.debug('creating shell channel queue')
            assert self.channel_socket is not None
            self._shell_channel = ChannelQueue('shell', self.channel_socket, self.log)
            assert self._channel_queues is not None
            self._channel_queues['shell'] = self._shell_channel
        return self._shell_channel

    @property
    def iopub_channel(self):
        """Get the iopub channel object for this kernel."""
        if self._iopub_channel is None:
            self.log.debug('creating iopub channel queue')
            assert self.channel_socket is not None
            self._iopub_channel = ChannelQueue('iopub', self.channel_socket, self.log)
            assert self._channel_queues is not None
            self._channel_queues['iopub'] = self._iopub_channel
        return self._iopub_channel

    @property
    def stdin_channel(self):
        """Get the stdin channel object for this kernel."""
        if self._stdin_channel is None:
            self.log.debug('creating stdin channel queue')
            assert self.channel_socket is not None
            self._stdin_channel = ChannelQueue('stdin', self.channel_socket, self.log)
            assert self._channel_queues is not None
            self._channel_queues['stdin'] = self._stdin_channel
        return self._stdin_channel

    @property
    def hb_channel(self):
        """Get the hb channel object for this kernel."""
        if self._hb_channel is None:
            self.log.debug('creating hb channel queue')
            assert self.channel_socket is not None
            self._hb_channel = HBChannelQueue('hb', self.channel_socket, self.log)
            assert self._channel_queues is not None
            self._channel_queues['hb'] = self._hb_channel
        return self._hb_channel

    @property
    def control_channel(self):
        """Get the control channel object for this kernel."""
        if self._control_channel is None:
            self.log.debug('creating control channel queue')
            assert self.channel_socket is not None
            self._control_channel = ChannelQueue('control', self.channel_socket, self.log)
            assert self._channel_queues is not None
            self._channel_queues['control'] = self._control_channel
        return self._control_channel

    def _route_responses(self):
        """
        Reads responses from the websocket and routes each to the appropriate channel queue based
        on the message's channel.  It does this for the duration of the class's lifetime until the
        channels are stopped, at which time the socket is closed (unblocking the router) and
        the thread terminates.  If shutdown happens to occur while processing a response (unlikely),
        termination takes place via the loop control boolean.
        """
        try:
            while not self._channels_stopped:
                assert self.channel_socket is not None
                raw_message = self.channel_socket.recv()
                if not raw_message:
                    break
                response_message = json_decode(utf8(raw_message))
                channel = response_message['channel']
                assert self._channel_queues is not None
                self._channel_queues[channel].put_nowait(response_message)
        except websocket.WebSocketConnectionClosedException:
            pass
        except BaseException as be:
            if not self._channels_stopped:
                self.log.warning(f'Unexpected exception encountered ({be})')
        assert self._channel_queues is not None
        for channel_queue in self._channel_queues.values():
            channel_queue.response_router_finished = True
        self.log.debug('Response router thread exiting...')