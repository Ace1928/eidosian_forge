import ast
import base64
import logging
import os
import pathlib
from glob import glob
from types import ModuleType
from bokeh.application import Application
from bokeh.application.handlers.document_lifecycle import (
from bokeh.application.handlers.function import FunctionHandler
from bokeh.command.subcommands.serve import Serve as _BkServe
from bokeh.command.util import build_single_handler_applications
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import EMPTY_LAYOUT
from bokeh.server.contexts import ApplicationContext
from tornado.ioloop import PeriodicCallback
from tornado.web import StaticFileHandler
from ..auth import BasicAuthProvider, OAuthProvider
from ..config import config
from ..io.document import _cleanup_doc
from ..io.liveness import LivenessHandler
from ..io.reload import record_modules, watch
from ..io.rest import REST_PROVIDERS
from ..io.server import INDEX_HTML, get_static_routes, set_curdoc
from ..io.state import state
from ..util import fullpath
class AdminApplicationContext(ApplicationContext):

    def __init__(self, application, unused_timeout=15000, **kwargs):
        super().__init__(application, **kwargs)
        self._unused_timeout = unused_timeout
        self._cleanup_cb = None

    async def cleanup_sessions(self):
        await self._cleanup_sessions(self._unused_timeout)

    def run_load_hook(self):
        self._cleanup_cb = PeriodicCallback(self.cleanup_sessions, self._unused_timeout)
        self._cleanup_cb.start()
        super().run_load_hook()

    def run_unload_hook(self):
        if self._cleanup_cb:
            self._cleanup_cb.stop()
        super().run_unload_hook()