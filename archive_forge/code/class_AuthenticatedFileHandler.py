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
class AuthenticatedFileHandler(JupyterHandler, web.StaticFileHandler):
    """static files should only be accessible when logged in"""
    auth_resource = 'contents'

    @property
    def content_security_policy(self) -> str:
        return super().content_security_policy + '; sandbox allow-scripts'

    @web.authenticated
    @authorized
    def head(self, path: str) -> Awaitable[None]:
        """Get the head response for a path."""
        self.check_xsrf_cookie()
        return super().head(path)

    @web.authenticated
    @authorized
    def get(self, path: str, **kwargs: Any) -> Awaitable[None]:
        """Get a file by path."""
        self.check_xsrf_cookie()
        if os.path.splitext(path)[1] == '.ipynb' or self.get_argument('download', None):
            name = path.rsplit('/', 1)[-1]
            self.set_attachment_header(name)
        return web.StaticFileHandler.get(self, path, **kwargs)

    def get_content_type(self) -> str:
        """Get the content type."""
        assert self.absolute_path is not None
        path = self.absolute_path.strip('/')
        if '/' in path:
            _, name = path.rsplit('/', 1)
        else:
            name = path
        if name.endswith('.ipynb'):
            return 'application/x-ipynb+json'
        else:
            cur_mime = mimetypes.guess_type(name)[0]
            if cur_mime == 'text/plain':
                return 'text/plain; charset=UTF-8'
            else:
                return super().get_content_type()

    def set_headers(self) -> None:
        """Set the headers."""
        super().set_headers()
        if 'v' not in self.request.arguments:
            self.add_header('Cache-Control', 'no-cache')

    def compute_etag(self) -> str | None:
        """Compute the etag."""
        return None

    def validate_absolute_path(self, root: str, absolute_path: str) -> str:
        """Validate and return the absolute path.

        Requires tornado 3.1

        Adding to tornado's own handling, forbids the serving of hidden files.
        """
        abs_path = super().validate_absolute_path(root, absolute_path)
        abs_root = os.path.abspath(root)
        assert abs_path is not None
        if not self.contents_manager.allow_hidden and is_hidden(abs_path, abs_root):
            self.log.info("Refusing to serve hidden file, via 404 Error, use flag 'ContentsManager.allow_hidden' to enable")
            raise web.HTTPError(404)
        return abs_path