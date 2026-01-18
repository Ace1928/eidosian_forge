from __future__ import annotations
import ast
import html
import json
import logging
import os
import pathlib
import re
import sys
import traceback
import urllib.parse as urlparse
from contextlib import contextmanager
from types import ModuleType
from typing import IO, Any, Callable
import bokeh.command.util
from bokeh.application.handlers.code import CodeHandler
from bokeh.application.handlers.code_runner import CodeRunner
from bokeh.application.handlers.handler import Handler, handle_exception
from bokeh.core.types import PathLike
from bokeh.document import Document
from bokeh.io.doc import curdoc, patch_curdoc, set_curdoc as bk_set_curdoc
from bokeh.util.dependencies import import_required
from ..config import config
from .mime_render import MIME_RENDERERS
from .profile import profile_ctx
from .reload import record_modules
from .state import state
class PanelCodeHandler(CodeHandler):
    """Modify Bokeh documents by creating Dashboard from code.

    Additionally this subclass adds support for the ability to:

    - Log session launch, load and destruction
    - Capture document_ready events to track when app is loaded.
    - Add profiling support
    - Ensure that state.curdoc is set
    - Reload the application module if autoreload is enabled
    - Track modules loaded during app execution to enable autoreloading
    """

    def __init__(self, *, source: str, filename: PathLike, argv: list[str]=[], package: ModuleType | None=None) -> None:
        Handler.__init__(self)
        self._runner = PanelCodeRunner(source, filename, argv, package=package)
        self._loggers = {}
        for f in PanelCodeHandler._io_functions:
            self._loggers[f] = self._make_io_logger(f)

    def modify_document(self, doc: 'Document'):
        if config.autoreload:
            path = self._runner.path
            argv = self._runner._argv
            handler = type(self)(filename=path, argv=argv)
            self._runner = handler._runner
        module = self._runner.new_module()
        if module is None:
            return
        doc.modules.add(module)
        run_app(self, module, doc)