from __future__ import annotations
import logging
import os
from functools import partial
from typing import TYPE_CHECKING
import bokeh.command.util
from bokeh.application import Application as BkApplication
from bokeh.application.handlers.directory import DirectoryHandler
from bokeh.application.handlers.document_lifecycle import (
from ..config import config
from .document import _destroy_document
from .handlers import MarkdownHandler, NotebookHandler, ScriptHandler
from .logging import LOG_SESSION_DESTROYED, LOG_SESSION_LAUNCHING
from .state import set_curdoc, state
def _log_session_destroyed(session_context):
    log.info(LOG_SESSION_DESTROYED, id(doc))