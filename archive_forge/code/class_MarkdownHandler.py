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
class MarkdownHandler(PanelCodeHandler):
    """Modify Bokeh documents by creating Dashboard from a Markdown file.
    """
    _logger_text = "%s: call to %s() ignored when running Markdown files with the 'panel' command."
    _origin = 'Markdown'

    def __init__(self, *args, **kwargs):
        """

        Keywords:
            filename (str) : a path to a Markdown (".md") file

        """
        if 'filename' not in kwargs:
            raise ValueError('Must pass a filename to Handler')
        filename = os.path.abspath(kwargs['filename'])
        with open(filename, encoding='utf-8') as f:
            code = extract_code(f)
        kwargs['source'] = code
        super().__init__(*args, **kwargs)