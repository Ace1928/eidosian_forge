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
class GatewaySessionManager(SessionManager):
    """A gateway session manager."""
    kernel_manager = Instance('jupyter_server.gateway.managers.GatewayMappingKernelManager')

    async def kernel_culled(self, kernel_id: str) -> bool:
        """Checks if the kernel is still considered alive and returns true if it's not found."""
        km: Optional[GatewayKernelManager] = None
        try:
            km = self.kernel_manager.get_kernel(kernel_id)
        except Exception:
            pass
        return km is None