from __future__ import annotations
import logging # isort:skip
import gc
import os
from pprint import pformat
from types import ModuleType
from typing import (
from urllib.parse import urljoin
from tornado.ioloop import PeriodicCallback
from tornado.web import Application as TornadoApplication, StaticFileHandler
from tornado.websocket import WebSocketClosedError
from ..application import Application
from ..document import Document
from ..model import Model
from ..resources import Resources
from ..settings import settings
from ..util.dependencies import import_optional
from ..util.strings import format_docstring
from ..util.tornado import fixup_windows_event_loop_policy
from .auth_provider import NullAuth
from .connection import ServerConnection
from .contexts import ApplicationContext
from .session import ServerSession
from .urls import per_app_patterns, toplevel_patterns
from .views.ico_handler import IcoHandler
from .views.root_handler import RootHandler
from .views.static_handler import StaticHandler
from .views.ws import WSHandler
def _log_stats(self) -> None:
    log.trace('Running stats log job')
    if log.getEffectiveLevel() > logging.DEBUG:
        return
    log.debug('[pid %d] %d clients connected', PID, len(self._clients))
    for app_path, app in self._applications.items():
        sessions = list(app.sessions)
        unused_count = 0
        for s in sessions:
            if s.connection_count == 0:
                unused_count += 1
        log.debug('[pid %d]   %s has %d sessions with %d unused', PID, app_path, len(sessions), unused_count)