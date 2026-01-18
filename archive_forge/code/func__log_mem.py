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
def _log_mem(self) -> None:
    mem = PROC.memory_info()
    log.info('[pid %d] Memory usage: %0.2f MB (RSS), %0.2f MB (VMS)', PID, mem.rss / GB, mem.vms / GB)
    del mem
    if log.getEffectiveLevel() > logging.DEBUG:
        return
    all_objs = gc.get_objects()
    for name, typ in [('Documents', Document), ('Sessions', ServerSession), ('Models', Model)]:
        objs = [x for x in all_objs if isinstance(x, typ)]
        log.debug(f'  uncollected {name}: {len(objs)}')
    objs = [x for x in gc.get_objects() if isinstance(x, ModuleType) and 'bokeh_app_' in str(x)]
    log.debug(f'  uncollected modules: {len(objs)}')
    import pandas as pd
    objs = [x for x in all_objs if isinstance(x, pd.DataFrame)]
    log.debug('  uncollected DataFrames: %d', len(objs))