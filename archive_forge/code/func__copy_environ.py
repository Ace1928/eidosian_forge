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
def _copy_environ(self, other: WSGIEnvironment) -> WSGIEnvironment:
    out = {**self.environ_base, **other}
    if self.preserve_context:
        out['werkzeug.debug.preserve_context'] = self._new_contexts.append
    return out