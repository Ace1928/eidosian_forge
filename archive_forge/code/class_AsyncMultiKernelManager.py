from __future__ import annotations
import asyncio
import json
import os
import socket
import typing as t
import uuid
from functools import wraps
from pathlib import Path
import zmq
from traitlets import Any, Bool, Dict, DottedObjectName, Instance, Unicode, default, observe
from traitlets.config.configurable import LoggingConfigurable
from traitlets.utils.importstring import import_item
from .connect import KernelConnectionInfo
from .kernelspec import NATIVE_KERNEL_NAME, KernelSpecManager
from .manager import KernelManager
from .utils import ensure_async, run_sync, utcnow
class AsyncMultiKernelManager(MultiKernelManager):
    kernel_manager_class = DottedObjectName('jupyter_client.ioloop.AsyncIOLoopKernelManager', config=True, help='The kernel manager class.  This is configurable to allow\n        subclassing of the AsyncKernelManager for customized behavior.\n        ')
    use_pending_kernels = Bool(False, help='Whether to make kernels available before the process has started.  The\n        kernel has a `.ready` future which can be awaited before connecting').tag(config=True)
    context = Instance('zmq.asyncio.Context')

    @default('context')
    def _context_default(self) -> zmq.asyncio.Context:
        self._created_context = True
        return zmq.asyncio.Context()
    start_kernel: t.Callable[..., t.Awaitable] = MultiKernelManager._async_start_kernel
    restart_kernel: t.Callable[..., t.Awaitable] = MultiKernelManager._async_restart_kernel
    shutdown_kernel: t.Callable[..., t.Awaitable] = MultiKernelManager._async_shutdown_kernel
    shutdown_all: t.Callable[..., t.Awaitable] = MultiKernelManager._async_shutdown_all