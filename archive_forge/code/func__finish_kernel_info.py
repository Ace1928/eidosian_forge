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
def _finish_kernel_info(self, info):
    """Finish handling kernel_info reply

        Set up protocol adaptation, if needed,
        and signal that connection can continue.
        """
    protocol_version = info.get('protocol_version', client_protocol_version)
    if protocol_version != client_protocol_version:
        self.session.adapt_version = int(protocol_version.split('.')[0])
        self.log.info(f'Adapting from protocol version {protocol_version} (kernel {self.kernel_id}) to {client_protocol_version} (client).')
    if not self._kernel_info_future.done():
        self._kernel_info_future.set_result(info)