from __future__ import annotations
import datetime
import errno
import gettext
import hashlib
import hmac
import ipaddress
import json
import logging
import mimetypes
import os
import pathlib
import random
import re
import select
import signal
import socket
import stat
import sys
import threading
import time
import typing as t
import urllib
import warnings
from base64 import encodebytes
from pathlib import Path
import jupyter_client
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager
from jupyter_client.session import Session
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from jupyter_core.paths import jupyter_runtime_dir
from jupyter_events.logger import EventLogger
from nbformat.sign import NotebookNotary
from tornado import httpserver, ioloop, web
from tornado.httputil import url_concat
from tornado.log import LogFormatter, access_log, app_log, gen_log
from tornado.netutil import bind_sockets
from tornado.routing import Matcher, Rule
from traitlets import (
from traitlets.config import Config
from traitlets.config.application import boolean_flag, catch_config_error
from jupyter_server import (
from jupyter_server._sysinfo import get_sys_info
from jupyter_server._tz import utcnow
from jupyter_server.auth.authorizer import AllowAllAuthorizer, Authorizer
from jupyter_server.auth.identity import (
from jupyter_server.auth.login import LoginHandler
from jupyter_server.auth.logout import LogoutHandler
from jupyter_server.base.handlers import (
from jupyter_server.extension.config import ExtensionConfigManager
from jupyter_server.extension.manager import ExtensionManager
from jupyter_server.extension.serverextension import ServerExtensionApp
from jupyter_server.gateway.connections import GatewayWebSocketConnection
from jupyter_server.gateway.gateway_client import GatewayClient
from jupyter_server.gateway.managers import (
from jupyter_server.log import log_request
from jupyter_server.services.config import ConfigManager
from jupyter_server.services.contents.filemanager import (
from jupyter_server.services.contents.largefilemanager import AsyncLargeFileManager
from jupyter_server.services.contents.manager import AsyncContentsManager, ContentsManager
from jupyter_server.services.kernels.connection.base import BaseKernelWebsocketConnection
from jupyter_server.services.kernels.connection.channels import ZMQChannelsWebsocketConnection
from jupyter_server.services.kernels.kernelmanager import (
from jupyter_server.services.sessions.sessionmanager import SessionManager
from jupyter_server.utils import (
from jinja2 import Environment, FileSystemLoader
from jupyter_core.paths import secure_write
from jupyter_core.utils import ensure_async
from jupyter_server.transutils import _i18n, trans
from jupyter_server.utils import pathname2url, urljoin
class ServerWebApplication(web.Application):
    """A server web application."""

    def __init__(self, jupyter_app, default_services, kernel_manager, contents_manager, session_manager, kernel_spec_manager, config_manager, event_logger, extra_services, log, base_url, default_url, settings_overrides, jinja_env_options, *, authorizer=None, identity_provider=None, kernel_websocket_connection_class=None, websocket_ping_interval=None, websocket_ping_timeout=None):
        """Initialize a server web application."""
        if identity_provider is None:
            warnings.warn('identity_provider unspecified. Using default IdentityProvider. Specify an identity_provider to avoid this message.', RuntimeWarning, stacklevel=2)
            identity_provider = IdentityProvider(parent=jupyter_app)
        if authorizer is None:
            warnings.warn('authorizer unspecified. Using permissive AllowAllAuthorizer. Specify an authorizer to avoid this message.', JupyterServerAuthWarning, stacklevel=2)
            authorizer = AllowAllAuthorizer(parent=jupyter_app, identity_provider=identity_provider)
        settings = self.init_settings(jupyter_app, kernel_manager, contents_manager, session_manager, kernel_spec_manager, config_manager, event_logger, extra_services, log, base_url, default_url, settings_overrides, jinja_env_options, authorizer=authorizer, identity_provider=identity_provider, kernel_websocket_connection_class=kernel_websocket_connection_class, websocket_ping_interval=websocket_ping_interval, websocket_ping_timeout=websocket_ping_timeout)
        handlers = self.init_handlers(default_services, settings)
        undecorated_methods = []
        for matcher, handler, *_ in handlers:
            undecorated_methods.extend(self._check_handler_auth(matcher, handler))
        if undecorated_methods:
            message = 'Core endpoints without @allow_unauthenticated, @ws_authenticated, nor @web.authenticated:\n' + '\n'.join(undecorated_methods)
            if jupyter_app.allow_unauthenticated_access:
                warnings.warn(message, JupyterServerAuthWarning, stacklevel=2)
            else:
                raise Exception(message)
        super().__init__(handlers, **settings)

    def add_handlers(self, host_pattern, host_handlers):
        undecorated_methods = []
        for rule in host_handlers:
            if isinstance(rule, Rule):
                matcher = rule.matcher
                handler = rule.target
            else:
                matcher, handler, *_ = rule
            undecorated_methods.extend(self._check_handler_auth(matcher, handler))
        if undecorated_methods and (not self.settings['allow_unauthenticated_access']):
            message = 'Extension endpoints without @allow_unauthenticated, @ws_authenticated, nor @web.authenticated:\n' + '\n'.join(undecorated_methods)
            warnings.warn(message, JupyterServerAuthWarning, stacklevel=2)
        return super().add_handlers(host_pattern, host_handlers)

    def init_settings(self, jupyter_app, kernel_manager, contents_manager, session_manager, kernel_spec_manager, config_manager, event_logger, extra_services, log, base_url, default_url, settings_overrides, jinja_env_options=None, *, authorizer=None, identity_provider=None, kernel_websocket_connection_class=None, websocket_ping_interval=None, websocket_ping_timeout=None):
        """Initialize settings for the web application."""
        _template_path = settings_overrides.get('template_path', jupyter_app.template_file_path)
        if isinstance(_template_path, str):
            _template_path = (_template_path,)
        template_path = [os.path.expanduser(path) for path in _template_path]
        jenv_opt: dict[str, t.Any] = {'autoescape': True}
        jenv_opt.update(jinja_env_options if jinja_env_options else {})
        env = Environment(loader=FileSystemLoader(template_path), extensions=['jinja2.ext.i18n'], **jenv_opt)
        sys_info = get_sys_info()
        base_dir = os.path.realpath(os.path.join(__file__, '..', '..'))
        nbui = gettext.translation('nbui', localedir=os.path.join(base_dir, 'jupyter_server/i18n'), fallback=True)
        env.install_gettext_translations(nbui, newstyle=False)
        if sys_info['commit_source'] == 'repository':
            version_hash = ''
        else:
            utc = datetime.timezone.utc
            version_hash = datetime.datetime.now(tz=utc).strftime('%Y%m%d%H%M%S')
        now = utcnow()
        root_dir = contents_manager.root_dir
        home = os.path.expanduser('~')
        if root_dir.startswith(home + os.path.sep):
            root_dir = '~' + root_dir[len(home):]
        settings = {'log_function': log_request, 'base_url': base_url, 'default_url': default_url, 'template_path': template_path, 'static_path': jupyter_app.static_file_path, 'static_custom_path': jupyter_app.static_custom_path, 'static_handler_class': FileFindHandler, 'static_url_prefix': url_path_join(base_url, '/static/'), 'static_handler_args': {'no_cache_paths': [url_path_join(base_url, 'static', 'custom')]}, 'version_hash': version_hash, 'kernel_ws_protocol': jupyter_app.kernel_ws_protocol, 'limit_rate': jupyter_app.limit_rate, 'iopub_msg_rate_limit': jupyter_app.iopub_msg_rate_limit, 'iopub_data_rate_limit': jupyter_app.iopub_data_rate_limit, 'rate_limit_window': jupyter_app.rate_limit_window, 'cookie_secret': jupyter_app.cookie_secret, 'login_url': url_path_join(base_url, '/login'), 'xsrf_cookies': True, 'disable_check_xsrf': jupyter_app.disable_check_xsrf, 'allow_unauthenticated_access': jupyter_app.allow_unauthenticated_access, 'allow_remote_access': jupyter_app.allow_remote_access, 'local_hostnames': jupyter_app.local_hostnames, 'authenticate_prometheus': jupyter_app.authenticate_prometheus, 'kernel_manager': kernel_manager, 'contents_manager': contents_manager, 'session_manager': session_manager, 'kernel_spec_manager': kernel_spec_manager, 'config_manager': config_manager, 'authorizer': authorizer, 'identity_provider': identity_provider, 'event_logger': event_logger, 'kernel_websocket_connection_class': kernel_websocket_connection_class, 'websocket_ping_interval': websocket_ping_interval, 'websocket_ping_timeout': websocket_ping_timeout, 'extra_services': extra_services, 'started': now, 'last_activity_times': {}, 'jinja_template_vars': jupyter_app.jinja_template_vars, 'websocket_url': jupyter_app.websocket_url, 'shutdown_button': jupyter_app.quit_button, 'config': jupyter_app.config, 'config_dir': jupyter_app.config_dir, 'allow_password_change': jupyter_app.allow_password_change, 'server_root_dir': root_dir, 'jinja2_env': env, 'serverapp': jupyter_app}
        settings.update(settings_overrides)
        if base_url and 'xsrf_cookie_kwargs' not in settings:
            settings['xsrf_cookie_kwargs'] = {'path': base_url}
        return settings

    def init_handlers(self, default_services, settings):
        """Load the (URL pattern, handler) tuples for each component."""
        handlers = []
        for service in settings['extra_services']:
            handlers.extend(load_handlers(service))
        for service in default_services:
            if service in JUPYTER_SERVICE_HANDLERS:
                locations = JUPYTER_SERVICE_HANDLERS[service]
                if locations is not None:
                    for loc in locations:
                        handlers.extend(load_handlers(loc))
            else:
                msg = f'{service} is not recognized as a jupyter_server service. If this is a custom service, try adding it to the `extra_services` list.'
                raise Exception(msg)
        handlers.extend(settings['contents_manager'].get_extra_handlers())
        handlers.extend(settings['identity_provider'].get_handlers())
        handlers.extend(load_handlers('jupyter_server.base.handlers'))
        if settings['default_url'] != settings['base_url']:
            handlers.append(('/?', RedirectWithParams, {'url': settings['default_url'], 'permanent': False}))
        else:
            handlers.append(('/', MainHandler))
        new_handlers = []
        for handler in handlers:
            pattern = url_path_join(settings['base_url'], handler[0])
            new_handler = (pattern, *list(handler[1:]))
            new_handlers.append(new_handler)
        new_handlers.append(('(.*)', Template404))
        return new_handlers

    def last_activity(self):
        """Get a UTC timestamp for when the server last did something.

        Includes: API activity, kernel activity, kernel shutdown, and terminal
        activity.
        """
        sources = [self.settings['started'], self.settings['kernel_manager'].last_kernel_activity]
        sources.extend([val for key, val in self.settings.items() if key.endswith('_last_activity')])
        sources.extend(self.settings['last_activity_times'].values())
        return max(sources)

    def _check_handler_auth(self, matcher: t.Union[str, Matcher], handler: type[web.RequestHandler]):
        missing_authentication = []
        for method_name in handler.SUPPORTED_METHODS:
            method = getattr(handler, method_name.lower())
            is_unimplemented = method == web.RequestHandler._unimplemented_method
            is_allowlisted = hasattr(method, '__allow_unauthenticated')
            is_blocklisted = _has_tornado_web_authenticated(method)
            if not is_unimplemented and (not is_allowlisted) and (not is_blocklisted):
                missing_authentication.append(f'- {method_name} of {handler.__name__} registered for {matcher}')
        return missing_authentication