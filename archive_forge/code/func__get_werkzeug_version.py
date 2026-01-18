from __future__ import annotations
import importlib.metadata
import typing as t
from contextlib import contextmanager
from contextlib import ExitStack
from copy import copy
from types import TracebackType
from urllib.parse import urlsplit
import werkzeug.test
from click.testing import CliRunner
from werkzeug.test import Client
from werkzeug.wrappers import Request as BaseRequest
from .cli import ScriptInfo
from .sessions import SessionMixin
def _get_werkzeug_version() -> str:
    global _werkzeug_version
    if not _werkzeug_version:
        _werkzeug_version = importlib.metadata.version('werkzeug')
    return _werkzeug_version