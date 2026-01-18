from __future__ import annotations
import itertools
import json
import os
import re
import typing as t
import warnings
from fnmatch import fnmatch
from jupyter_core.utils import ensure_async, run_sync
from jupyter_events import EventLogger
from nbformat import ValidationError, sign
from nbformat import validate as validate_nb
from nbformat.v4 import new_notebook
from tornado.web import HTTPError, RequestHandler
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
from jupyter_server.transutils import _i18n
from jupyter_server.utils import import_item
from ...files.handlers import FilesHandler
from .checkpoints import AsyncCheckpoints, Checkpoints
@validate('pre_save_hook')
def _validate_pre_save_hook(self, proposal):
    value = proposal['value']
    if isinstance(value, str):
        value = import_item(self.pre_save_hook)
    if not callable(value):
        msg = 'pre_save_hook must be callable'
        raise TraitError(msg)
    if callable(self.pre_save_hook):
        warnings.warn(f'Overriding existing pre_save_hook ({self.pre_save_hook.__name__}) with a new one ({value.__name__}).', stacklevel=2)
    return value