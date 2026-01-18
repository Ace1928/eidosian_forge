from __future__ import annotations
import functools
import inspect
import ipaddress
import json
import mimetypes
import os
import re
import types
import warnings
from http.client import responses
from logging import Logger
from typing import TYPE_CHECKING, Any, Awaitable, Coroutine, Sequence, cast
from urllib.parse import urlparse
import prometheus_client
from jinja2 import TemplateNotFound
from jupyter_core.paths import is_hidden
from jupyter_events import EventLogger
from tornado import web
from tornado.log import app_log
from traitlets.config import Application
import jupyter_server
from jupyter_server import CallContext
from jupyter_server._sysinfo import get_sys_info
from jupyter_server._tz import utcnow
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.auth.identity import User
from jupyter_server.i18n import combine_translations
from jupyter_server.services.security import csp_report_uri
from jupyter_server.utils import (
class FileFindHandler(JupyterHandler, web.StaticFileHandler):
    """subclass of StaticFileHandler for serving files from a search path

    The setting "static_immutable_cache" can be set up to serve some static
    file as immutable (e.g. file name containing a hash). The setting is a
    list of base URL, every static file URL starting with one of those will
    be immutable.
    """
    _static_paths: dict[str, str] = {}
    root: tuple[str]

    def set_headers(self) -> None:
        """Set the headers."""
        super().set_headers()
        immutable_paths = self.settings.get('static_immutable_cache', [])
        if any((self.request.path.startswith(path) for path in immutable_paths)):
            self.set_header('Cache-Control', 'public, max-age=31536000, immutable')
        elif 'v' not in self.request.arguments or any((self.request.path.startswith(path) for path in self.no_cache_paths)):
            self.set_header('Cache-Control', 'no-cache')

    def initialize(self, path: str | list[str], default_filename: str | None=None, no_cache_paths: list[str] | None=None) -> None:
        """Initialize the file find handler."""
        self.no_cache_paths = no_cache_paths or []
        if isinstance(path, str):
            path = [path]
        self.root = tuple((os.path.abspath(os.path.expanduser(p)) + os.sep for p in path))
        self.default_filename = default_filename

    def compute_etag(self) -> str | None:
        """Compute the etag."""
        return None

    @allow_unauthenticated
    def get(self, path: str, include_body: bool=True) -> Coroutine[Any, Any, None]:
        return super().get(path, include_body)

    @allow_unauthenticated
    def head(self, path: str) -> Awaitable[None]:
        return super().head(path)

    @classmethod
    def get_absolute_path(cls, roots: Sequence[str], path: str) -> str:
        """locate a file to serve on our static file search path"""
        with cls._lock:
            if path in cls._static_paths:
                return cls._static_paths[path]
            try:
                abspath = os.path.abspath(filefind(path, roots))
            except OSError:
                return ''
            cls._static_paths[path] = abspath
            log().debug(f'Path {path} served from {abspath}')
            return abspath

    def validate_absolute_path(self, root: str, absolute_path: str) -> str | None:
        """check if the file should be served (raises 404, 403, etc.)"""
        if not absolute_path:
            raise web.HTTPError(404)
        for root in self.root:
            if (absolute_path + os.sep).startswith(root):
                break
        return super().validate_absolute_path(root, absolute_path)