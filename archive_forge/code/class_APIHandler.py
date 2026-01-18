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
class APIHandler(JupyterHandler):
    """Base class for API handlers"""

    async def prepare(self) -> None:
        """Prepare an API response."""
        await super().prepare()
        if not self.check_origin():
            raise web.HTTPError(404)

    def write_error(self, status_code: int, **kwargs: Any) -> None:
        """APIHandler errors are JSON, not human pages"""
        self.set_header('Content-Type', 'application/json')
        message = responses.get(status_code, 'Unknown HTTP Error')
        reply: dict[str, Any] = {'message': message}
        exc_info = kwargs.get('exc_info')
        if exc_info:
            e = exc_info[1]
            if isinstance(e, HTTPError):
                reply['message'] = e.log_message or message
                reply['reason'] = e.reason
            else:
                reply['message'] = 'Unhandled error'
                reply['reason'] = None
                reply['traceback'] = ''
        self.log.warning('wrote error: %r', reply['message'], exc_info=True)
        self.finish(json.dumps(reply))

    def get_login_url(self) -> str:
        """Get the login url."""
        if not self.current_user:
            raise web.HTTPError(403)
        return super().get_login_url()

    @property
    def content_security_policy(self) -> str:
        csp = '; '.join([super().content_security_policy, "default-src 'none'"])
        return csp
    _track_activity = True

    def update_api_activity(self) -> None:
        """Update last_activity of API requests"""
        if self._track_activity and getattr(self, '_jupyter_current_user', None) and (self.get_argument('no_track_activity', None) is None):
            self.settings['api_last_activity'] = utcnow()

    def finish(self, *args: Any, **kwargs: Any) -> Future[Any]:
        """Finish an API response."""
        self.update_api_activity()
        set_content_type = kwargs.pop('set_content_type', 'application/json')
        self.set_header('Content-Type', set_content_type)
        return super().finish(*args, **kwargs)

    @allow_unauthenticated
    def options(self, *args: Any, **kwargs: Any) -> None:
        """Get the options."""
        if 'Access-Control-Allow-Headers' in self.settings.get('headers', {}):
            self.set_header('Access-Control-Allow-Headers', self.settings['headers']['Access-Control-Allow-Headers'])
        else:
            self.set_header('Access-Control-Allow-Headers', 'accept, content-type, authorization, x-xsrftoken')
        self.set_header('Access-Control-Allow-Methods', 'GET, PUT, POST, PATCH, DELETE, OPTIONS')
        requested_headers = self.request.headers.get('Access-Control-Request-Headers', '').split(',')
        if requested_headers and any((h.strip().lower() == 'authorization' for h in requested_headers)) and self.login_available and (self.allow_origin or self.allow_origin_pat or 'Access-Control-Allow-Origin' in self.settings.get('headers', {})):
            self.set_header('Access-Control-Allow-Origin', self.request.headers.get('Origin', ''))