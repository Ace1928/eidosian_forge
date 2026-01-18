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
class ServerKernelManager(AsyncIOLoopKernelManager):
    """A server-specific kernel manager."""
    execution_state = Unicode(None, allow_none=True, help='The current execution state of the kernel')
    reason = Unicode('', help='The reason for the last failure against the kernel')
    last_activity = Instance(datetime, help='The last activity on the kernel')

    @property
    def core_event_schema_paths(self) -> list[pathlib.Path]:
        return [DEFAULT_EVENTS_SCHEMA_PATH / 'kernel_actions' / 'v1.yaml']
    extra_event_schema_paths: List[str] = List(default_value=[], help="\n        A list of pathlib.Path objects pointing at to register with\n        the kernel manager's eventlogger.\n        ").tag(config=True)
    event_logger = Instance(EventLogger)

    @default('event_logger')
    def _default_event_logger(self):
        """Initialize the logger and ensure all required events are present."""
        if self.parent is not None and self.parent.parent is not None and hasattr(self.parent.parent, 'event_logger'):
            logger = self.parent.parent.event_logger
        else:
            logger = EventLogger()
        schemas = self.core_event_schema_paths + self.extra_event_schema_paths
        for schema_path in schemas:
            try:
                logger.register_event_schema(schema_path)
            except SchemaRegistryException:
                pass
        return logger

    def emit(self, schema_id, data):
        """Emit an event from the kernel manager."""
        self.event_logger.emit(schema_id=schema_id, data=data)

    @overrides
    @emit_kernel_action_event(success_msg='Kernel {kernel_id} was started.')
    async def start_kernel(self, *args, **kwargs):
        return await super().start_kernel(*args, **kwargs)

    @overrides
    @emit_kernel_action_event(success_msg='Kernel {kernel_id} was shutdown.')
    async def shutdown_kernel(self, *args, **kwargs):
        return await super().shutdown_kernel(*args, **kwargs)

    @overrides
    @emit_kernel_action_event(success_msg='Kernel {kernel_id} was restarted.')
    async def restart_kernel(self, *args, **kwargs):
        return await super().restart_kernel(*args, **kwargs)

    @overrides
    @emit_kernel_action_event(success_msg='Kernel {kernel_id} was interrupted.')
    async def interrupt_kernel(self, *args, **kwargs):
        return await super().interrupt_kernel(*args, **kwargs)