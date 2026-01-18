from __future__ import annotations
import asyncio
import os
import pathlib
import typing as t
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial, wraps
from jupyter_client.ioloop.manager import AsyncIOLoopKernelManager
from jupyter_client.multikernelmanager import AsyncMultiKernelManager, MultiKernelManager
from jupyter_client.session import Session
from jupyter_core.paths import exists
from jupyter_core.utils import ensure_async
from jupyter_events import EventLogger
from jupyter_events.schema_registry import SchemaRegistryException
from overrides import overrides
from tornado import web
from tornado.concurrent import Future
from tornado.ioloop import IOLoop, PeriodicCallback
from traitlets import (
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH
from jupyter_server._tz import isoformat, utcnow
from jupyter_server.prometheus.metrics import KERNEL_CURRENTLY_RUNNING_TOTAL
from jupyter_server.utils import ApiPath, import_item, to_os_path
class AsyncMappingKernelManager(MappingKernelManager, AsyncMultiKernelManager):
    """An asynchronous mapping kernel manager."""

    @default('kernel_manager_class')
    def _default_kernel_manager_class(self):
        return 'jupyter_server.services.kernels.kernelmanager.ServerKernelManager'

    @validate('kernel_manager_class')
    def _validate_kernel_manager_class(self, proposal):
        """A validator for the kernel manager class."""
        km_class_value = proposal.value
        km_class = import_item(km_class_value)
        if not issubclass(km_class, ServerKernelManager):
            warnings.warn(f"KernelManager class '{km_class}' is not a subclass of 'ServerKernelManager'.  Custom KernelManager classes should derive from 'ServerKernelManager' beginning with jupyter-server 2.0 or risk missing functionality.  Continuing...", FutureWarning, stacklevel=3)
        return km_class_value

    def __init__(self, **kwargs):
        """Initialize an async mapping kernel manager."""
        self.pinned_superclass = MultiKernelManager
        self._pending_kernel_tasks = {}
        self.pinned_superclass.__init__(self, **kwargs)
        self.last_kernel_activity = utcnow()