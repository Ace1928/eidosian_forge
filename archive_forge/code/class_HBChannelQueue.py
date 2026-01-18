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
class HBChannelQueue(ChannelQueue):
    """A queue for the heartbeat channel."""

    def is_beating(self) -> bool:
        """Whether the channel is beating."""
        return self.is_alive()