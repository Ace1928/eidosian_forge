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
def _handle_kernel_info_reply(self, msg):
    """process the kernel_info_reply

        enabling msg spec adaptation, if necessary
        """
    idents, msg = self.session.feed_identities(msg)
    try:
        msg = self.session.deserialize(msg)
    except BaseException:
        self.log.error('Bad kernel_info reply', exc_info=True)
        self._kernel_info_future.set_result({})
        return
    else:
        info = msg['content']
        self.log.debug('Received kernel info: %s', info)
        if msg['msg_type'] != 'kernel_info_reply' or 'protocol_version' not in info:
            self.log.error('Kernel info request failed, assuming current %s', info)
            info = {}
        self._finish_kernel_info(info)
    if self.kernel_info_channel:
        self.kernel_info_channel.close()
    self.kernel_info_channel = None