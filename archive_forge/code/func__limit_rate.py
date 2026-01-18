from __future__ import annotations
import asyncio
import json
import time
import typing as t
import weakref
from concurrent.futures import Future
from textwrap import dedent
from jupyter_client import protocol_version as client_protocol_version  # type:ignore[attr-defined]
from tornado import gen, web
from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketClosedError
from traitlets import Any, Bool, Dict, Float, Instance, Int, List, Unicode, default
from jupyter_core.utils import ensure_async
from jupyter_server.transutils import _i18n
from ..websocket import KernelWebsocketHandler
from .abc import KernelWebsocketConnectionABC
from .base import (
def _limit_rate(self, channel, msg, msg_list):
    """Limit the message rate on a channel."""
    if not (self.limit_rate and channel == 'iopub'):
        return False
    msg['header'] = self.get_part('header', msg['header'], msg_list)
    msg_type = msg['header']['msg_type']
    if msg_type == 'status':
        msg['content'] = self.get_part('content', msg['content'], msg_list)
        if msg['content'].get('execution_state') == 'idle':
            self._iopub_window_byte_queue = []
            self._iopub_window_msg_count = 0
            self._iopub_window_byte_count = 0
            self._iopub_msgs_exceeded = False
            self._iopub_data_exceeded = False
    if msg_type not in {'status', 'comm_open', 'execute_input'}:
        now = IOLoop.current().time()
        while len(self._iopub_window_byte_queue) > 0:
            queued = self._iopub_window_byte_queue[0]
            if now >= queued[0]:
                self._iopub_window_byte_count -= queued[1]
                self._iopub_window_msg_count -= 1
                del self._iopub_window_byte_queue[0]
            else:
                break
        self._iopub_window_msg_count += 1
        byte_count = sum((len(x) for x in msg_list)) if msg_type == 'stream' else 0
        self._iopub_window_byte_count += byte_count
        self._iopub_window_byte_queue.append((now + self.rate_limit_window, byte_count))
        msg_rate = float(self._iopub_window_msg_count) / self.rate_limit_window
        data_rate = float(self._iopub_window_byte_count) / self.rate_limit_window
        if self.iopub_msg_rate_limit > 0 and msg_rate > self.iopub_msg_rate_limit:
            if not self._iopub_msgs_exceeded:
                self._iopub_msgs_exceeded = True
                msg['parent_header'] = self.get_part('parent_header', msg['parent_header'], msg_list)
                self.write_stderr(dedent(f'                    IOPub message rate exceeded.\n                    The Jupyter server will temporarily stop sending output\n                    to the client in order to avoid crashing it.\n                    To change this limit, set the config variable\n                    `--ServerApp.iopub_msg_rate_limit`.\n\n                    Current values:\n                    ServerApp.iopub_msg_rate_limit={self.iopub_msg_rate_limit} (msgs/sec)\n                    ServerApp.rate_limit_window={self.rate_limit_window} (secs)\n                    '), msg['parent_header'])
        elif self._iopub_msgs_exceeded and msg_rate < 0.8 * self.iopub_msg_rate_limit:
            self._iopub_msgs_exceeded = False
            if not self._iopub_data_exceeded:
                self.log.warning('iopub messages resumed')
        if self.iopub_data_rate_limit > 0 and data_rate > self.iopub_data_rate_limit:
            if not self._iopub_data_exceeded:
                self._iopub_data_exceeded = True
                msg['parent_header'] = self.get_part('parent_header', msg['parent_header'], msg_list)
                self.write_stderr(dedent(f'                    IOPub data rate exceeded.\n                    The Jupyter server will temporarily stop sending output\n                    to the client in order to avoid crashing it.\n                    To change this limit, set the config variable\n                    `--ServerApp.iopub_data_rate_limit`.\n\n                    Current values:\n                    ServerApp.iopub_data_rate_limit={self.iopub_data_rate_limit} (bytes/sec)\n                    ServerApp.rate_limit_window={self.rate_limit_window} (secs)\n                    '), msg['parent_header'])
        elif self._iopub_data_exceeded and data_rate < 0.8 * self.iopub_data_rate_limit:
            self._iopub_data_exceeded = False
            if not self._iopub_msgs_exceeded:
                self.log.warning('iopub messages resumed')
        if self._iopub_msgs_exceeded or self._iopub_data_exceeded:
            self._iopub_window_msg_count -= 1
            self._iopub_window_byte_count -= byte_count
            self._iopub_window_byte_queue.pop(-1)
            return True
        return False