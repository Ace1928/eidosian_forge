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
class AuthenticatedHandler(web.RequestHandler):
    """A RequestHandler with an authenticated user."""

    @property
    def base_url(self) -> str:
        return cast(str, self.settings.get('base_url', '/'))

    @property
    def content_security_policy(self) -> str:
        """The default Content-Security-Policy header

        Can be overridden by defining Content-Security-Policy in settings['headers']
        """
        if 'Content-Security-Policy' in self.settings.get('headers', {}):
            return cast(str, self.settings['headers']['Content-Security-Policy'])
        return '; '.join(["frame-ancestors 'self'", 'report-uri ' + self.settings.get('csp_report_uri', url_path_join(self.base_url, csp_report_uri))])

    def set_default_headers(self) -> None:
        """Set the default headers."""
        headers = {}
        headers['X-Content-Type-Options'] = 'nosniff'
        headers.update(self.settings.get('headers', {}))
        headers['Content-Security-Policy'] = self.content_security_policy
        for header_name, value in headers.items():
            try:
                self.set_header(header_name, value)
            except Exception as e:
                self.log.exception('Could not set default headers: %s', e)

    @property
    def cookie_name(self) -> str:
        warnings.warn('JupyterHandler.login_handler is deprecated in 2.0,\n            use JupyterHandler.identity_provider.\n            ', DeprecationWarning, stacklevel=2)
        return self.identity_provider.get_cookie_name(self)

    def force_clear_cookie(self, name: str, path: str='/', domain: str | None=None) -> None:
        """Force a cookie clear."""
        warnings.warn('JupyterHandler.login_handler is deprecated in 2.0,\n            use JupyterHandler.identity_provider.\n            ', DeprecationWarning, stacklevel=2)
        self.identity_provider._force_clear_cookie(self, name, path=path, domain=domain)

    def clear_login_cookie(self) -> None:
        """Clear a login cookie."""
        warnings.warn('JupyterHandler.login_handler is deprecated in 2.0,\n            use JupyterHandler.identity_provider.\n            ', DeprecationWarning, stacklevel=2)
        self.identity_provider.clear_login_cookie(self)

    def get_current_user(self) -> str:
        """Get the current user."""
        clsname = self.__class__.__name__
        msg = f'Calling `{clsname}.get_current_user()` directly is deprecated in jupyter-server 2.0. Use `self.current_user` instead (works in all versions).'
        if hasattr(self, '_jupyter_current_user'):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return cast(str, self._jupyter_current_user)
        raise RuntimeError(msg)

    def skip_check_origin(self) -> bool:
        """Ask my login_handler if I should skip the origin_check

        For example: in the default LoginHandler, if a request is token-authenticated,
        origin checking should be skipped.
        """
        if self.request.method == 'OPTIONS':
            return True
        return not self.identity_provider.should_check_origin(self)

    @property
    def token_authenticated(self) -> bool:
        """Have I been authenticated with a token?"""
        return self.identity_provider.is_token_authenticated(self)

    @property
    def logged_in(self) -> bool:
        """Is a user currently logged in?"""
        user = self.current_user
        return bool(user and user != 'anonymous')

    @property
    def login_handler(self) -> Any:
        """Return the login handler for this application, if any."""
        warnings.warn('JupyterHandler.login_handler is deprecated in 2.0,\n            use JupyterHandler.identity_provider.\n            ', DeprecationWarning, stacklevel=2)
        return self.identity_provider.login_handler_class

    @property
    def token(self) -> str | None:
        """Return the login token for this application, if any."""
        return self.identity_provider.token

    @property
    def login_available(self) -> bool:
        """May a user proceed to log in?

        This returns True if login capability is available, irrespective of
        whether the user is already logged in or not.

        """
        return cast(bool, self.identity_provider.login_available)

    @property
    def authorizer(self) -> Authorizer:
        if 'authorizer' not in self.settings:
            warnings.warn("The Tornado web application does not have an 'authorizer' defined in its settings. In future releases of jupyter_server, this will be a required key for all subclasses of `JupyterHandler`. For an example, see the jupyter_server source code for how to add an authorizer to the tornado settings: https://github.com/jupyter-server/jupyter_server/blob/653740cbad7ce0c8a8752ce83e4d3c2c754b13cb/jupyter_server/serverapp.py#L234-L256", stacklevel=2)
            from jupyter_server.auth import AllowAllAuthorizer
            self.settings['authorizer'] = AllowAllAuthorizer(config=self.settings.get('config', None), identity_provider=self.identity_provider)
        return cast('Authorizer', self.settings.get('authorizer'))

    @property
    def identity_provider(self) -> IdentityProvider:
        if 'identity_provider' not in self.settings:
            warnings.warn("The Tornado web application does not have an 'identity_provider' defined in its settings. In future releases of jupyter_server, this will be a required key for all subclasses of `JupyterHandler`. For an example, see the jupyter_server source code for how to add an identity provider to the tornado settings: https://github.com/jupyter-server/jupyter_server/blob/v2.0.0/jupyter_server/serverapp.py#L242", stacklevel=2)
            from jupyter_server.auth import IdentityProvider
            self.settings['identity_provider'] = IdentityProvider(config=self.settings.get('config', None))
        return cast('IdentityProvider', self.settings['identity_provider'])