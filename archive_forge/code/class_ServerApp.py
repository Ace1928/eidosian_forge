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
class ServerApp(JupyterApp):
    """The Jupyter Server application class."""
    name = 'jupyter-server'
    version: str = __version__
    description: str = _i18n('The Jupyter Server.\n\n    This launches a Tornado-based Jupyter Server.')
    examples = _examples
    flags = Dict(flags)
    aliases = Dict(aliases)
    classes = [KernelManager, Session, MappingKernelManager, KernelSpecManager, AsyncMappingKernelManager, ContentsManager, FileContentsManager, AsyncContentsManager, AsyncFileContentsManager, NotebookNotary, GatewayMappingKernelManager, GatewayKernelSpecManager, GatewaySessionManager, GatewayWebSocketConnection, GatewayClient, Authorizer, EventLogger, ZMQChannelsWebsocketConnection]
    subcommands: dict[str, t.Any] = {'list': (JupyterServerListApp, JupyterServerListApp.description.splitlines()[0]), 'stop': (JupyterServerStopApp, JupyterServerStopApp.description.splitlines()[0]), 'password': (JupyterPasswordApp, JupyterPasswordApp.description.splitlines()[0]), 'extension': (ServerExtensionApp, ServerExtensionApp.description.splitlines()[0])}
    default_services = ('api', 'auth', 'config', 'contents', 'files', 'kernels', 'kernelspecs', 'nbconvert', 'security', 'sessions', 'shutdown', 'view', 'events')
    _log_formatter_cls = LogFormatter
    _stopping = Bool(False, help="Signal that we've begun stopping.")

    @default('log_level')
    def _default_log_level(self) -> int:
        return logging.INFO

    @default('log_format')
    def _default_log_format(self) -> str:
        """override default log format to include date & time"""
        return '%(color)s[%(levelname)1.1s %(asctime)s.%(msecs).03d %(name)s]%(end_color)s %(message)s'
    file_to_run = Unicode('', help='Open the named file when the application is launched.').tag(config=True)
    file_url_prefix = Unicode('notebooks', help='The URL prefix where files are opened directly.').tag(config=True)
    allow_origin = Unicode('', config=True, help="Set the Access-Control-Allow-Origin header\n\n        Use '*' to allow any origin to access your server.\n\n        Takes precedence over allow_origin_pat.\n        ")
    allow_origin_pat = Unicode('', config=True, help='Use a regular expression for the Access-Control-Allow-Origin header\n\n        Requests from an origin matching the expression will get replies with:\n\n            Access-Control-Allow-Origin: origin\n\n        where `origin` is the origin of the request.\n\n        Ignored if allow_origin is set.\n        ')
    allow_credentials = Bool(False, config=True, help=_i18n('Set the Access-Control-Allow-Credentials: true header'))
    allow_root = Bool(False, config=True, help=_i18n('Whether to allow the user to run the server as root.'))
    autoreload = Bool(False, config=True, help=_i18n('Reload the webapp when changes are made to any Python src files.'))
    default_url = Unicode('/', config=True, help=_i18n('The default URL to redirect to from `/`'))
    ip = Unicode('localhost', config=True, help=_i18n('The IP address the Jupyter server will listen on.'))

    @default('ip')
    def _default_ip(self) -> str:
        """Return localhost if available, 127.0.0.1 otherwise.

        On some (horribly broken) systems, localhost cannot be bound.
        """
        s = socket.socket()
        try:
            s.bind(('localhost', 0))
        except OSError as e:
            self.log.warning(_i18n('Cannot bind to localhost, using 127.0.0.1 as default ip\n%s'), e)
            return '127.0.0.1'
        else:
            s.close()
            return 'localhost'

    @validate('ip')
    def _validate_ip(self, proposal: t.Any) -> str:
        value = t.cast(str, proposal['value'])
        if value == '*':
            value = ''
        return value
    custom_display_url = Unicode('', config=True, help=_i18n('Override URL shown to users.\n\n        Replace actual URL, including protocol, address, port and base URL,\n        with the given value when displaying URL to the users. Do not change\n        the actual connection URL. If authentication token is enabled, the\n        token is added to the custom URL automatically.\n\n        This option is intended to be used when the URL to display to the user\n        cannot be determined reliably by the Jupyter server (proxified\n        or containerized setups for example).'))
    port_env = 'JUPYTER_PORT'
    port_default_value = DEFAULT_JUPYTER_SERVER_PORT
    port = Integer(config=True, help=_i18n('The port the server will listen on (env: JUPYTER_PORT).'))

    @default('port')
    def _port_default(self) -> int:
        return int(os.getenv(self.port_env, self.port_default_value))
    port_retries_env = 'JUPYTER_PORT_RETRIES'
    port_retries_default_value = 50
    port_retries = Integer(port_retries_default_value, config=True, help=_i18n('The number of additional ports to try if the specified port is not available (env: JUPYTER_PORT_RETRIES).'))

    @default('port_retries')
    def _port_retries_default(self) -> int:
        return int(os.getenv(self.port_retries_env, self.port_retries_default_value))
    sock = Unicode('', config=True, help='The UNIX socket the Jupyter server will listen on.')
    sock_mode = Unicode('0600', config=True, help='The permissions mode for UNIX socket creation (default: 0600).')

    @validate('sock_mode')
    def _validate_sock_mode(self, proposal: t.Any) -> t.Any:
        value = proposal['value']
        try:
            converted_value = int(value.encode(), 8)
            assert all((bool(converted_value & stat.S_IRUSR), bool(converted_value & stat.S_IWUSR), converted_value <= 2 ** 12))
        except ValueError as e:
            raise TraitError('invalid --sock-mode value: %s, please specify as e.g. "0600"' % value) from e
        except AssertionError as e:
            raise TraitError('invalid --sock-mode value: %s, must have u+rw (0600) at a minimum' % value) from e
        return value
    certfile = Unicode('', config=True, help=_i18n('The full path to an SSL/TLS certificate file.'))
    keyfile = Unicode('', config=True, help=_i18n('The full path to a private key file for usage with SSL/TLS.'))
    client_ca = Unicode('', config=True, help=_i18n('The full path to a certificate authority certificate for SSL/TLS client authentication.'))
    cookie_secret_file = Unicode(config=True, help=_i18n('The file where the cookie secret is stored.'))

    @default('cookie_secret_file')
    def _default_cookie_secret_file(self) -> str:
        return os.path.join(self.runtime_dir, 'jupyter_cookie_secret')
    cookie_secret = Bytes(b'', config=True, help='The random bytes used to secure cookies.\n        By default this is a new random number every time you start the server.\n        Set it to a value in a config file to enable logins to persist across server sessions.\n\n        Note: Cookie secrets should be kept private, do not share config files with\n        cookie_secret stored in plaintext (you can read the value from a file).\n        ')

    @default('cookie_secret')
    def _default_cookie_secret(self) -> bytes:
        if os.path.exists(self.cookie_secret_file):
            with open(self.cookie_secret_file, 'rb') as f:
                key = f.read()
        else:
            key = encodebytes(os.urandom(32))
            self._write_cookie_secret_file(key)
        h = hmac.new(key, digestmod=hashlib.sha256)
        h.update(self.password.encode())
        return h.digest()

    def _write_cookie_secret_file(self, secret: bytes) -> None:
        """write my secret to my secret_file"""
        self.log.info(_i18n('Writing Jupyter server cookie secret to %s'), self.cookie_secret_file)
        try:
            with secure_write(self.cookie_secret_file, True) as f:
                f.write(secret)
        except OSError as e:
            self.log.error(_i18n('Failed to write cookie secret to %s: %s'), self.cookie_secret_file, e)
    _token_set = False
    token = Unicode('<DEPRECATED>', help=_i18n('DEPRECATED. Use IdentityProvider.token')).tag(config=True)

    @observe('token')
    def _deprecated_token(self, change: t.Any) -> None:
        self._warn_deprecated_config(change, 'IdentityProvider')

    @default('token')
    def _deprecated_token_access(self) -> str:
        warnings.warn('ServerApp.token config is deprecated in jupyter-server 2.0. Use IdentityProvider.token', DeprecationWarning, stacklevel=3)
        return self.identity_provider.token
    min_open_files_limit = Integer(config=True, help='\n        Gets or sets a lower bound on the open file handles process resource\n        limit. This may need to be increased if you run into an\n        OSError: [Errno 24] Too many open files.\n        This is not applicable when running on Windows.\n        ', allow_none=True)

    @default('min_open_files_limit')
    def _default_min_open_files_limit(self) -> t.Optional[int]:
        if resource is None:
            return None
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        DEFAULT_SOFT = 4096
        if hard >= DEFAULT_SOFT:
            return DEFAULT_SOFT
        self.log.debug('Default value for min_open_files_limit is ignored (hard=%r, soft=%r)', hard, soft)
        return soft
    max_body_size = Integer(512 * 1024 * 1024, config=True, help='\n        Sets the maximum allowed size of the client request body, specified in\n        the Content-Length request header field. If the size in a request\n        exceeds the configured value, a malformed HTTP message is returned to\n        the client.\n\n        Note: max_body_size is applied even in streaming mode.\n        ')
    max_buffer_size = Integer(512 * 1024 * 1024, config=True, help='\n        Gets or sets the maximum amount of memory, in bytes, that is allocated\n        for use by the buffer manager.\n        ')
    password = Unicode('', config=True, help='DEPRECATED in 2.0. Use PasswordIdentityProvider.hashed_password')
    password_required = Bool(False, config=True, help='DEPRECATED in 2.0. Use PasswordIdentityProvider.password_required')
    allow_password_change = Bool(True, config=True, help='DEPRECATED in 2.0. Use PasswordIdentityProvider.allow_password_change')

    def _warn_deprecated_config(self, change: t.Any, clsname: str, new_name: t.Optional[str]=None) -> None:
        """Warn on deprecated config."""
        if new_name is None:
            new_name = change.name
        if clsname not in self.config or new_name not in self.config[clsname]:
            self.log.warning(f'ServerApp.{change.name} config is deprecated in 2.0. Use {clsname}.{new_name}.')
            self.config[clsname][new_name] = change.new
        elif self.config[clsname][new_name] != change.new:
            self.log.warning(f'Ignoring deprecated ServerApp.{change.name} config. Using {clsname}.{new_name}.')

    @observe('password')
    def _deprecated_password(self, change: t.Any) -> None:
        self._warn_deprecated_config(change, 'PasswordIdentityProvider', new_name='hashed_password')

    @observe('password_required', 'allow_password_change')
    def _deprecated_password_config(self, change: t.Any) -> None:
        self._warn_deprecated_config(change, 'PasswordIdentityProvider')
    disable_check_xsrf = Bool(False, config=True, help='Disable cross-site-request-forgery protection\n\n        Jupyter server includes protection from cross-site request forgeries,\n        requiring API requests to either:\n\n        - originate from pages served by this server (validated with XSRF cookie and token), or\n        - authenticate with a token\n\n        Some anonymous compute resources still desire the ability to run code,\n        completely without authentication.\n        These services can disable all authentication and security checks,\n        with the full knowledge of what that implies.\n        ')
    _allow_unauthenticated_access_env = 'JUPYTER_SERVER_ALLOW_UNAUTHENTICATED_ACCESS'
    allow_unauthenticated_access = Bool(True, config=True, help=f'Allow unauthenticated access to endpoints without authentication rule.\n\n        When set to `True` (default in jupyter-server 2.0, subject to change\n        in the future), any request to an endpoint without an authentication rule\n        (either `@tornado.web.authenticated`, or `@allow_unauthenticated`)\n        will be permitted, regardless of whether user has logged in or not.\n\n        When set to `False`, logging in will be required for access to each endpoint,\n        excluding the endpoints marked with `@allow_unauthenticated` decorator.\n\n        This option can be configured using `{_allow_unauthenticated_access_env}`\n        environment variable: any non-empty value other than "true" and "yes" will\n        prevent unauthenticated access to endpoints without `@allow_unauthenticated`.\n        ')

    @default('allow_unauthenticated_access')
    def _allow_unauthenticated_access_default(self):
        if os.getenv(self._allow_unauthenticated_access_env):
            return os.environ[self._allow_unauthenticated_access_env].lower() in ['true', 'yes']
        return True
    allow_remote_access = Bool(config=True, help="Allow requests where the Host header doesn't point to a local server\n\n       By default, requests get a 403 forbidden response if the 'Host' header\n       shows that the browser thinks it's on a non-local domain.\n       Setting this option to True disables this check.\n\n       This protects against 'DNS rebinding' attacks, where a remote web server\n       serves you a page and then changes its DNS to send later requests to a\n       local IP, bypassing same-origin checks.\n\n       Local IP addresses (such as 127.0.0.1 and ::1) are allowed as local,\n       along with hostnames configured in local_hostnames.\n       ")

    @default('allow_remote_access')
    def _default_allow_remote(self) -> bool:
        """Disallow remote access if we're listening only on loopback addresses"""
        if self.ip == '':
            return True
        try:
            addr = ipaddress.ip_address(self.ip)
        except ValueError:
            for info in socket.getaddrinfo(self.ip, self.port, 0, socket.SOCK_STREAM):
                addr = info[4][0]
                try:
                    parsed = ipaddress.ip_address(addr.split('%')[0])
                except ValueError:
                    self.log.warning('Unrecognised IP address: %r', addr)
                    continue
                if not (parsed.is_loopback or ('%' in addr and parsed.is_link_local)):
                    return True
            return False
        else:
            return not addr.is_loopback
    use_redirect_file = Bool(True, config=True, help='Disable launching browser by redirect file\n     For versions of notebook > 5.7.2, a security feature measure was added that\n     prevented the authentication token used to launch the browser from being visible.\n     This feature makes it difficult for other users on a multi-user system from\n     running code in your Jupyter session as you.\n     However, some environments (like Windows Subsystem for Linux (WSL) and Chromebooks),\n     launching a browser using a redirect file can lead the browser failing to load.\n     This is because of the difference in file structures/paths between the runtime and\n     the browser.\n\n     Disabling this setting to False will disable this behavior, allowing the browser\n     to launch by using a URL and visible token (as before).\n     ')
    local_hostnames = List(Unicode(), ['localhost'], config=True, help='Hostnames to allow as local when allow_remote_access is False.\n\n       Local IP addresses (such as 127.0.0.1 and ::1) are automatically accepted\n       as local as well.\n       ')
    open_browser = Bool(False, config=True, help='Whether to open in a browser after starting.\n                        The specific browser used is platform dependent and\n                        determined by the python standard library `webbrowser`\n                        module, unless it is overridden using the --browser\n                        (ServerApp.browser) configuration option.\n                        ')
    browser = Unicode('', config=True, help='Specify what command to use to invoke a web\n                      browser when starting the server. If not specified, the\n                      default browser will be determined by the `webbrowser`\n                      standard library module, which allows setting of the\n                      BROWSER environment variable to override it.\n                      ')
    webbrowser_open_new = Integer(2, config=True, help=_i18n('Specify where to open the server on startup. This is the\n        `new` argument passed to the standard library method `webbrowser.open`.\n        The behaviour is not guaranteed, but depends on browser support. Valid\n        values are:\n\n         - 2 opens a new tab,\n         - 1 opens a new window,\n         - 0 opens in an existing window.\n\n        See the `webbrowser.open` documentation for details.\n        '))
    tornado_settings = Dict(config=True, help=_i18n('Supply overrides for the tornado.web.Application that the Jupyter server uses.'))
    websocket_compression_options = Any(None, config=True, help=_i18n('\n        Set the tornado compression options for websocket connections.\n\n        This value will be returned from :meth:`WebSocketHandler.get_compression_options`.\n        None (default) will disable compression.\n        A dict (even an empty one) will enable compression.\n\n        See the tornado docs for WebSocketHandler.get_compression_options for details.\n        '))
    terminado_settings = Dict(Union([List(), Unicode()]), config=True, help=_i18n('Supply overrides for terminado. Currently only supports "shell_command".'))
    cookie_options = Dict(config=True, help=_i18n('DEPRECATED. Use IdentityProvider.cookie_options'))
    get_secure_cookie_kwargs = Dict(config=True, help=_i18n('DEPRECATED. Use IdentityProvider.get_secure_cookie_kwargs'))

    @observe('cookie_options', 'get_secure_cookie_kwargs')
    def _deprecated_cookie_config(self, change: t.Any) -> None:
        self._warn_deprecated_config(change, 'IdentityProvider')
    ssl_options = Dict(allow_none=True, config=True, help=_i18n('Supply SSL options for the tornado HTTPServer.\n            See the tornado docs for details.'))
    jinja_environment_options = Dict(config=True, help=_i18n('Supply extra arguments that will be passed to Jinja environment.'))
    jinja_template_vars = Dict(config=True, help=_i18n('Extra variables to supply to jinja templates when rendering.'))
    base_url = Unicode('/', config=True, help='The base URL for the Jupyter server.\n\n                       Leading and trailing slashes can be omitted,\n                       and will automatically be added.\n                       ')

    @validate('base_url')
    def _update_base_url(self, proposal: t.Any) -> str:
        value = t.cast(str, proposal['value'])
        if not value.startswith('/'):
            value = '/' + value
        if not value.endswith('/'):
            value = value + '/'
        return value
    extra_static_paths = List(Unicode(), config=True, help='Extra paths to search for serving static files.\n\n        This allows adding javascript/css to be available from the Jupyter server machine,\n        or overriding individual files in the IPython')

    @property
    def static_file_path(self) -> list[str]:
        """return extra paths + the default location"""
        return [*self.extra_static_paths, DEFAULT_STATIC_FILES_PATH]
    static_custom_path = List(Unicode(), help=_i18n('Path to search for custom.js, css'))

    @default('static_custom_path')
    def _default_static_custom_path(self) -> list[str]:
        return [os.path.join(d, 'custom') for d in (self.config_dir, DEFAULT_STATIC_FILES_PATH)]
    extra_template_paths = List(Unicode(), config=True, help=_i18n('Extra paths to search for serving jinja templates.\n\n        Can be used to override templates from jupyter_server.templates.'))

    @property
    def template_file_path(self) -> list[str]:
        """return extra paths + the default locations"""
        return self.extra_template_paths + DEFAULT_TEMPLATE_PATH_LIST
    extra_services = List(Unicode(), config=True, help=_i18n('handlers that should be loaded at higher priority than the default services'))
    websocket_url = Unicode('', config=True, help="The base URL for websockets,\n        if it differs from the HTTP server (hint: it almost certainly doesn't).\n\n        Should be in the form of an HTTP origin: ws[s]://hostname[:port]\n        ")
    quit_button = Bool(True, config=True, help='If True, display controls to shut down the Jupyter server, such as menu items or buttons.')
    contents_manager_class = Type(default_value=AsyncLargeFileManager, klass=ContentsManager, config=True, help=_i18n('The content manager class to use.'))
    kernel_manager_class = Type(klass=MappingKernelManager, config=True, help=_i18n('The kernel manager class to use.'))

    @default('kernel_manager_class')
    def _default_kernel_manager_class(self) -> t.Union[str, type[AsyncMappingKernelManager]]:
        if self.gateway_config.gateway_enabled:
            return 'jupyter_server.gateway.managers.GatewayMappingKernelManager'
        return AsyncMappingKernelManager
    session_manager_class = Type(config=True, help=_i18n('The session manager class to use.'))

    @default('session_manager_class')
    def _default_session_manager_class(self) -> t.Union[str, type[SessionManager]]:
        if self.gateway_config.gateway_enabled:
            return 'jupyter_server.gateway.managers.GatewaySessionManager'
        return SessionManager
    kernel_websocket_connection_class = Type(klass=BaseKernelWebsocketConnection, config=True, help=_i18n('The kernel websocket connection class to use.'))

    @default('kernel_websocket_connection_class')
    def _default_kernel_websocket_connection_class(self) -> t.Union[str, type[ZMQChannelsWebsocketConnection]]:
        if self.gateway_config.gateway_enabled:
            return 'jupyter_server.gateway.connections.GatewayWebSocketConnection'
        return ZMQChannelsWebsocketConnection
    websocket_ping_interval = Integer(config=True, help='\n            Configure the websocket ping interval in seconds.\n\n            Websockets are long-lived connections that are used by some Jupyter\n            Server extensions.\n\n            Periodic pings help to detect disconnected clients and keep the\n            connection active. If this is set to None, then no pings will be\n            performed.\n\n            When a ping is sent, the client has ``websocket_ping_timeout``\n            seconds to respond. If no response is received within this period,\n            the connection will be closed from the server side.\n        ')
    websocket_ping_timeout = Integer(config=True, help='\n            Configure the websocket ping timeout in seconds.\n\n            See ``websocket_ping_interval`` for details.\n        ')
    config_manager_class = Type(default_value=ConfigManager, config=True, help=_i18n('The config manager class to use'))
    kernel_spec_manager = Instance(KernelSpecManager, allow_none=True)
    kernel_spec_manager_class = Type(config=True, help='\n        The kernel spec manager class to use. Should be a subclass\n        of `jupyter_client.kernelspec.KernelSpecManager`.\n\n        The Api of KernelSpecManager is provisional and might change\n        without warning between this version of Jupyter and the next stable one.\n        ')

    @default('kernel_spec_manager_class')
    def _default_kernel_spec_manager_class(self) -> t.Union[str, type[KernelSpecManager]]:
        if self.gateway_config.gateway_enabled:
            return 'jupyter_server.gateway.managers.GatewayKernelSpecManager'
        return KernelSpecManager
    login_handler_class = Type(default_value=LoginHandler, klass=web.RequestHandler, allow_none=True, config=True, help=_i18n('The login handler class to use.'))
    logout_handler_class = Type(default_value=LogoutHandler, klass=web.RequestHandler, allow_none=True, config=True, help=_i18n('The logout handler class to use.'))
    authorizer_class = Type(default_value=AllowAllAuthorizer, klass=Authorizer, config=True, help=_i18n('The authorizer class to use.'))
    identity_provider_class = Type(default_value=PasswordIdentityProvider, klass=IdentityProvider, config=True, help=_i18n('The identity provider class to use.'))
    trust_xheaders = Bool(False, config=True, help=_i18n('Whether to trust or not X-Scheme/X-Forwarded-Proto and X-Real-Ip/X-Forwarded-For headerssent by the upstream reverse proxy. Necessary if the proxy handles SSL'))
    event_logger = Instance(EventLogger, allow_none=True, help='An EventLogger for emitting structured event data from Jupyter Server and extensions.')
    info_file = Unicode()

    @default('info_file')
    def _default_info_file(self) -> str:
        info_file = 'jpserver-%s.json' % os.getpid()
        return os.path.join(self.runtime_dir, info_file)
    no_browser_open_file = Bool(False, help='If True, do not write redirect HTML file disk, or show in messages.')
    browser_open_file = Unicode()

    @default('browser_open_file')
    def _default_browser_open_file(self) -> str:
        basename = 'jpserver-%s-open.html' % os.getpid()
        return os.path.join(self.runtime_dir, basename)
    browser_open_file_to_run = Unicode()

    @default('browser_open_file_to_run')
    def _default_browser_open_file_to_run(self) -> str:
        basename = 'jpserver-file-to-run-%s-open.html' % os.getpid()
        return os.path.join(self.runtime_dir, basename)
    pylab = Unicode('disabled', config=True, help=_i18n('\n        DISABLED: use %pylab or %matplotlib in the notebook to enable matplotlib.\n        '))

    @observe('pylab')
    def _update_pylab(self, change: t.Any) -> None:
        """when --pylab is specified, display a warning and exit"""
        backend = ' %s' % change['new'] if change['new'] != 'warn' else ''
        self.log.error(_i18n('Support for specifying --pylab on the command line has been removed.'))
        self.log.error(_i18n('Please use `%pylab{0}` or `%matplotlib{0}` in the notebook itself.').format(backend))
        self.exit(1)
    notebook_dir = Unicode(config=True, help=_i18n('DEPRECATED, use root_dir.'))

    @observe('notebook_dir')
    def _update_notebook_dir(self, change: t.Any) -> None:
        if self._root_dir_set:
            return
        self.log.warning(_i18n('notebook_dir is deprecated, use root_dir'))
        self.root_dir = change['new']
    external_connection_dir = Unicode(None, allow_none=True, config=True, help=_i18n('The directory to look at for external kernel connection files, if allow_external_kernels is True. Defaults to Jupyter runtime_dir/external_kernels. Make sure that this directory is not filled with left-over connection files, that could result in unnecessary kernel manager creations.'))
    allow_external_kernels = Bool(False, config=True, help=_i18n('Whether or not to allow external kernels, whose connection files are placed in external_connection_dir.'))
    root_dir = Unicode(config=True, help=_i18n('The directory to use for notebooks and kernels.'))
    _root_dir_set = False

    @default('root_dir')
    def _default_root_dir(self) -> str:
        if self.file_to_run:
            self._root_dir_set = True
            return os.path.dirname(os.path.abspath(self.file_to_run))
        else:
            return os.getcwd()

    def _normalize_dir(self, value: str) -> str:
        """Normalize a directory."""
        _, path = os.path.splitdrive(value)
        if path == os.sep:
            return value
        value = value.rstrip(os.sep)
        if not os.path.isabs(value):
            value = os.path.abspath(value)
        return value

    @validate('root_dir')
    def _root_dir_validate(self, proposal: t.Any) -> str:
        value = self._normalize_dir(proposal['value'])
        if not os.path.isdir(value):
            raise TraitError(trans.gettext("No such directory: '%r'") % value)
        return value

    @observe('root_dir')
    def _root_dir_changed(self, change: t.Any) -> None:
        self._root_dir_set = True
    preferred_dir = Unicode(config=True, help=trans.gettext('Preferred starting directory to use for notebooks and kernels. ServerApp.preferred_dir is deprecated in jupyter-server 2.0. Use FileContentsManager.preferred_dir instead'))

    @default('preferred_dir')
    def _default_prefered_dir(self) -> str:
        return self.root_dir

    @validate('preferred_dir')
    def _preferred_dir_validate(self, proposal: t.Any) -> str:
        value = self._normalize_dir(proposal['value'])
        if not os.path.isdir(value):
            raise TraitError(trans.gettext("No such preferred dir: '%r'") % value)
        return value

    @observe('server_extensions')
    def _update_server_extensions(self, change: t.Any) -> None:
        self.log.warning(_i18n('server_extensions is deprecated, use jpserver_extensions'))
        self.server_extensions = change['new']
    jpserver_extensions = Dict(default_value={}, value_trait=Bool(), config=True, help=_i18n('Dict of Python modules to load as Jupyter server extensions.Entry values can be used to enable and disable the loading ofthe extensions. The extensions will be loaded in alphabetical order.'))
    reraise_server_extension_failures = Bool(False, config=True, help=_i18n('Reraise exceptions encountered loading server extensions?'))
    kernel_ws_protocol = Unicode(allow_none=True, config=True, help=_i18n('DEPRECATED. Use ZMQChannelsWebsocketConnection.kernel_ws_protocol'))

    @observe('kernel_ws_protocol')
    def _deprecated_kernel_ws_protocol(self, change: t.Any) -> None:
        self._warn_deprecated_config(change, 'ZMQChannelsWebsocketConnection')
    limit_rate = Bool(allow_none=True, config=True, help=_i18n('DEPRECATED. Use ZMQChannelsWebsocketConnection.limit_rate'))

    @observe('limit_rate')
    def _deprecated_limit_rate(self, change: t.Any) -> None:
        self._warn_deprecated_config(change, 'ZMQChannelsWebsocketConnection')
    iopub_msg_rate_limit = Float(allow_none=True, config=True, help=_i18n('DEPRECATED. Use ZMQChannelsWebsocketConnection.iopub_msg_rate_limit'))

    @observe('iopub_msg_rate_limit')
    def _deprecated_iopub_msg_rate_limit(self, change: t.Any) -> None:
        self._warn_deprecated_config(change, 'ZMQChannelsWebsocketConnection')
    iopub_data_rate_limit = Float(allow_none=True, config=True, help=_i18n('DEPRECATED. Use ZMQChannelsWebsocketConnection.iopub_data_rate_limit'))

    @observe('iopub_data_rate_limit')
    def _deprecated_iopub_data_rate_limit(self, change: t.Any) -> None:
        self._warn_deprecated_config(change, 'ZMQChannelsWebsocketConnection')
    rate_limit_window = Float(allow_none=True, config=True, help=_i18n('DEPRECATED. Use ZMQChannelsWebsocketConnection.rate_limit_window'))

    @observe('rate_limit_window')
    def _deprecated_rate_limit_window(self, change: t.Any) -> None:
        self._warn_deprecated_config(change, 'ZMQChannelsWebsocketConnection')
    shutdown_no_activity_timeout = Integer(0, config=True, help="Shut down the server after N seconds with no kernelsrunning and no activity. This can be used together with culling idle kernels (MappingKernelManager.cull_idle_timeout) to shutdown the Jupyter server when it's not in use. This is not precisely timed: it may shut down up to a minute later. 0 (the default) disables this automatic shutdown.")
    terminals_enabled = Bool(config=True, help=_i18n('Set to False to disable terminals.\n\n         This does *not* make the server more secure by itself.\n         Anything the user can in a terminal, they can also do in a notebook.\n\n         Terminals may also be automatically disabled if the terminado package\n         is not available.\n         '))

    @default('terminals_enabled')
    def _default_terminals_enabled(self) -> bool:
        return True
    authenticate_prometheus = Bool(True, help='"\n        Require authentication to access prometheus metrics.\n        ', config=True)
    static_immutable_cache = List(Unicode(), help='\n        Paths to set up static files as immutable.\n\n        This allow setting up the cache control of static files as immutable.\n        It should be used for static file named with a hash for instance.\n        ', config=True)
    _starter_app = Instance(default_value=None, allow_none=True, klass='jupyter_server.extension.application.ExtensionApp')

    @property
    def starter_app(self) -> t.Any:
        """Get the Extension that started this server."""
        return self._starter_app

    def parse_command_line(self, argv: t.Optional[list[str]]=None) -> None:
        """Parse the command line options."""
        super().parse_command_line(argv)
        if self.extra_args:
            arg0 = self.extra_args[0]
            f = os.path.abspath(arg0)
            self.argv.remove(arg0)
            if not os.path.exists(f):
                self.log.critical(_i18n('No such file or directory: %s'), f)
                self.exit(1)
            c = Config()
            if os.path.isdir(f):
                c.ServerApp.root_dir = f
            elif os.path.isfile(f):
                c.ServerApp.file_to_run = f
            self.update_config(c)

    def init_configurables(self) -> None:
        """Initialize configurables."""
        self.gateway_config = GatewayClient.instance(parent=self)
        if not issubclass(self.kernel_manager_class, AsyncMappingKernelManager):
            warnings.warn('The synchronous MappingKernelManager class is deprecated and will not be supported in Jupyter Server 3.0', DeprecationWarning, stacklevel=2)
        if not issubclass(self.contents_manager_class, AsyncContentsManager):
            warnings.warn('The synchronous ContentsManager classes are deprecated and will not be supported in Jupyter Server 3.0', DeprecationWarning, stacklevel=2)
        self.kernel_spec_manager = self.kernel_spec_manager_class(parent=self)
        kwargs = {'parent': self, 'log': self.log, 'connection_dir': self.runtime_dir, 'kernel_spec_manager': self.kernel_spec_manager}
        if jupyter_client.version_info > (8, 3, 0):
            if self.allow_external_kernels:
                external_connection_dir = self.external_connection_dir
                if external_connection_dir is None:
                    external_connection_dir = str(Path(self.runtime_dir) / 'external_kernels')
                kwargs['external_connection_dir'] = external_connection_dir
        elif self.allow_external_kernels:
            self.log.warning("Although allow_external_kernels=True, external kernels are not supported because jupyter-client's version does not allow them (should be >8.3.0).")
        self.kernel_manager = self.kernel_manager_class(**kwargs)
        self.contents_manager = self.contents_manager_class(parent=self, log=self.log)
        self.contents_manager.preferred_dir
        self.session_manager = self.session_manager_class(parent=self, log=self.log, kernel_manager=self.kernel_manager, contents_manager=self.contents_manager)
        self.config_manager = self.config_manager_class(parent=self, log=self.log)
        identity_provider_kwargs = {'parent': self, 'log': self.log}
        if self.login_handler_class is not LoginHandler and self.identity_provider_class is PasswordIdentityProvider:
            self.identity_provider_class = LegacyIdentityProvider
            self.log.warning(f'Customizing authentication via ServerApp.login_handler_class={self.login_handler_class} is deprecated in Jupyter Server 2.0. Use ServerApp.identity_provider_class. Falling back on legacy authentication.')
            identity_provider_kwargs['login_handler_class'] = self.login_handler_class
            if self.logout_handler_class:
                identity_provider_kwargs['logout_handler_class'] = self.logout_handler_class
        elif self.login_handler_class is not LoginHandler:
            self.log.warning(f'Ignoring deprecated config ServerApp.login_handler_class={self.login_handler_class}. Superseded by ServerApp.identity_provider_class={{self.identity_provider_class}}.')
        self.identity_provider = self.identity_provider_class(**identity_provider_kwargs)
        if self.identity_provider_class is LegacyIdentityProvider:
            self.tornado_settings['password'] = self.identity_provider.hashed_password
            self.tornado_settings['token'] = self.identity_provider.token
        if self._token_set:
            self.log.warning('ServerApp.token config is deprecated in jupyter-server 2.0. Use IdentityProvider.token')
            if self.identity_provider.token_generated:
                self.identity_provider.token_generated = False
                self.identity_provider.token = self.token
            else:
                self.log.warning('Ignoring deprecated ServerApp.token config')
        self.authorizer = self.authorizer_class(parent=self, log=self.log, identity_provider=self.identity_provider)

    def init_logging(self) -> None:
        """Initialize logging."""
        self.log.propagate = False
        for log in (app_log, access_log, gen_log):
            log.name = self.log.name
        logger = logging.getLogger('tornado')
        logger.propagate = True
        logger.parent = self.log
        logger.setLevel(self.log.level)

    def init_event_logger(self) -> None:
        """Initialize the Event Bus."""
        self.event_logger = EventLogger(parent=self)
        schema_ids = ['https://events.jupyter.org/jupyter_server/contents_service/v1', 'https://events.jupyter.org/jupyter_server/gateway_client/v1', 'https://events.jupyter.org/jupyter_server/kernel_actions/v1']
        for schema_id in schema_ids:
            rel_schema_path = schema_id.replace(JUPYTER_SERVER_EVENTS_URI + '/', '') + '.yaml'
            schema_path = DEFAULT_EVENTS_SCHEMA_PATH / rel_schema_path
            self.event_logger.register_event_schema(schema_path)

    def init_webapp(self) -> None:
        """initialize tornado webapp"""
        self.tornado_settings['allow_origin'] = self.allow_origin
        self.tornado_settings['websocket_compression_options'] = self.websocket_compression_options
        if self.allow_origin_pat:
            self.tornado_settings['allow_origin_pat'] = re.compile(self.allow_origin_pat)
        self.tornado_settings['allow_credentials'] = self.allow_credentials
        self.tornado_settings['autoreload'] = self.autoreload
        self.tornado_settings['cookie_options'] = self.identity_provider.cookie_options
        self.tornado_settings['get_secure_cookie_kwargs'] = self.identity_provider.get_secure_cookie_kwargs
        self.tornado_settings['token'] = self.identity_provider.token
        if self.static_immutable_cache:
            self.tornado_settings['static_immutable_cache'] = self.static_immutable_cache
        if not self.default_url.startswith(self.base_url):
            self.default_url = url_path_join(self.base_url, self.default_url)
        if self.sock:
            if self.port != DEFAULT_JUPYTER_SERVER_PORT:
                self.log.critical('Options --port and --sock are mutually exclusive. Aborting.')
                sys.exit(1)
            else:
                self.port = 0
            if self.open_browser:
                self.log.info('Ignoring --ServerApp.open_browser due to --sock being used.')
            if self.file_to_run:
                self.log.critical('Options --ServerApp.file_to_run and --sock are mutually exclusive.')
                sys.exit(1)
            if sys.platform.startswith('win'):
                self.log.critical('Option --sock is not supported on Windows, but got value of %s. Aborting.' % self.sock)
                sys.exit(1)
        self.web_app = ServerWebApplication(self, self.default_services, self.kernel_manager, self.contents_manager, self.session_manager, self.kernel_spec_manager, self.config_manager, self.event_logger, self.extra_services, self.log, self.base_url, self.default_url, self.tornado_settings, self.jinja_environment_options, authorizer=self.authorizer, identity_provider=self.identity_provider, kernel_websocket_connection_class=self.kernel_websocket_connection_class, websocket_ping_interval=self.websocket_ping_interval, websocket_ping_timeout=self.websocket_ping_timeout)
        if self.certfile:
            self.ssl_options['certfile'] = self.certfile
        if self.keyfile:
            self.ssl_options['keyfile'] = self.keyfile
        if self.client_ca:
            self.ssl_options['ca_certs'] = self.client_ca
        if not self.ssl_options:
            self.ssl_options = None
        else:
            import ssl
            self.ssl_options.setdefault('ssl_version', getattr(ssl, 'PROTOCOL_TLS', ssl.PROTOCOL_SSLv23))
            if self.ssl_options.get('ca_certs', False):
                self.ssl_options.setdefault('cert_reqs', ssl.CERT_REQUIRED)
        self.identity_provider.validate_security(self, ssl_options=self.ssl_options)
        if isinstance(self.identity_provider, LegacyIdentityProvider):
            self.identity_provider.settings = self.web_app.settings

    def init_resources(self) -> None:
        """initialize system resources"""
        if resource is None:
            self.log.debug('Ignoring min_open_files_limit because the limit cannot be adjusted (for example, on Windows)')
            return
        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        soft = self.min_open_files_limit
        hard = old_hard
        if soft is not None and old_soft < soft:
            if hard < soft:
                hard = soft
            self.log.debug(f'Raising open file limit: soft {old_soft}->{soft}; hard {old_hard}->{hard}')
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    def _get_urlparts(self, path: t.Optional[str]=None, include_token: bool=False) -> urllib.parse.ParseResult:
        """Constructs a urllib named tuple, ParseResult,
        with default values set by server config.
        The returned tuple can be manipulated using the `_replace` method.
        """
        if self.sock:
            scheme = 'http+unix'
            netloc = urlencode_unix_socket_path(self.sock)
        else:
            if not self.ip:
                ip = 'localhost'
            elif self.ip in ('0.0.0.0', '::'):
                ip = '%s' % socket.gethostname()
            else:
                ip = f'[{self.ip}]' if ':' in self.ip else self.ip
            netloc = f'{ip}:{self.port}'
            scheme = 'https' if self.certfile else 'http'
        if not path:
            path = self.default_url
        query = None
        if include_token and self.identity_provider.token:
            token = self.identity_provider.token if self.identity_provider.token_generated else '...'
            query = urllib.parse.urlencode({'token': token})
        urlparts = urllib.parse.ParseResult(scheme=scheme, netloc=netloc, path=path, query=query or '', params='', fragment='')
        return urlparts

    @property
    def public_url(self) -> str:
        parts = self._get_urlparts(include_token=True)
        if self.custom_display_url:
            custom = urllib.parse.urlparse(self.custom_display_url)._asdict()
            custom_updates = {key: item for key, item in custom.items() if item}
            parts = parts._replace(**custom_updates)
        return parts.geturl()

    @property
    def local_url(self) -> str:
        parts = self._get_urlparts(include_token=True)
        if not self.sock:
            parts = parts._replace(netloc=f'127.0.0.1:{self.port}')
        return parts.geturl()

    @property
    def display_url(self) -> str:
        """Human readable string with URLs for interacting
        with the running Jupyter Server
        """
        url = self.public_url + '\n    ' + self.local_url
        return url

    @property
    def connection_url(self) -> str:
        urlparts = self._get_urlparts(path=self.base_url)
        return urlparts.geturl()

    def init_signal(self) -> None:
        """Initialize signal handlers."""
        if not sys.platform.startswith('win') and sys.stdin and sys.stdin.isatty():
            signal.signal(signal.SIGINT, self._handle_sigint)
        signal.signal(signal.SIGTERM, self._signal_stop)
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self._signal_info)
        if hasattr(signal, 'SIGINFO'):
            signal.signal(signal.SIGINFO, self._signal_info)

    def _handle_sigint(self, sig: t.Any, frame: t.Any) -> None:
        """SIGINT handler spawns confirmation dialog"""
        signal.signal(signal.SIGINT, self._signal_stop)
        thread = threading.Thread(target=self._confirm_exit)
        thread.daemon = True
        thread.start()

    def _restore_sigint_handler(self) -> None:
        """callback for restoring original SIGINT handler"""
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _confirm_exit(self) -> None:
        """confirm shutdown on ^C

        A second ^C, or answering 'y' within 5s will cause shutdown,
        otherwise original SIGINT handler will be restored.

        This doesn't work on Windows.
        """
        info = self.log.info
        info(_i18n('interrupted'))
        if self.answer_yes:
            self.log.critical(_i18n('Shutting down...'))
            self.stop(from_signal=True)
            return
        info(self.running_server_info())
        yes = _i18n('y')
        no = _i18n('n')
        sys.stdout.write(_i18n('Shut down this Jupyter server (%s/[%s])? ') % (yes, no))
        sys.stdout.flush()
        r, w, x = select.select([sys.stdin], [], [], 5)
        if r:
            line = sys.stdin.readline()
            if line.lower().startswith(yes) and no not in line.lower():
                self.log.critical(_i18n('Shutdown confirmed'))
                self.stop(from_signal=True)
                return
        else:
            if self._stopping:
                return
            info(_i18n('No answer for 5s:'))
        info(_i18n('resuming operation...'))
        self.io_loop.add_callback_from_signal(self._restore_sigint_handler)

    def _signal_stop(self, sig: t.Any, frame: t.Any) -> None:
        """Handle a stop signal."""
        self.log.critical(_i18n('received signal %s, stopping'), sig)
        self.stop(from_signal=True)

    def _signal_info(self, sig: t.Any, frame: t.Any) -> None:
        """Handle an info signal."""
        self.log.info(self.running_server_info())

    def init_components(self) -> None:
        """Check the components submodule, and warn if it's unclean"""

    def find_server_extensions(self) -> None:
        """
        Searches Jupyter paths for jpserver_extensions.
        """
        manager = ExtensionConfigManager(read_config_path=self.config_file_paths)
        extensions = manager.get_jpserver_extensions()
        for modulename, enabled in sorted(extensions.items()):
            if modulename not in self.jpserver_extensions:
                self.config.ServerApp.jpserver_extensions.update({modulename: enabled})
                self.jpserver_extensions.update({modulename: enabled})

    def init_server_extensions(self) -> None:
        """
        If an extension's metadata includes an 'app' key,
        the value must be a subclass of ExtensionApp. An instance
        of the class will be created at this step. The config for
        this instance will inherit the ServerApp's config object
        and load its own config.
        """
        self.extension_manager = ExtensionManager(log=self.log, serverapp=self)
        self.extension_manager.from_jpserver_extensions(self.jpserver_extensions)
        self.extension_manager.link_all_extensions()

    def load_server_extensions(self) -> None:
        """Load any extensions specified by config.

        Import the module, then call the load_jupyter_server_extension function,
        if one exists.

        The extension API is experimental, and may change in future releases.
        """
        self.extension_manager.load_all_extensions()

    def init_mime_overrides(self) -> None:
        if os.name == 'nt':
            mimetypes.init(files=[])
        mimetypes.add_type('text/css', '.css')
        mimetypes.add_type('application/javascript', '.js')
        mimetypes.add_type('application/wasm', '.wasm')

    def shutdown_no_activity(self) -> None:
        """Shutdown server on timeout when there are no kernels or terminals."""
        km = self.kernel_manager
        if len(km) != 0:
            return
        if self.extension_manager.any_activity():
            return
        seconds_since_active = (utcnow() - self.web_app.last_activity()).total_seconds()
        self.log.debug('No activity for %d seconds.', seconds_since_active)
        if seconds_since_active > self.shutdown_no_activity_timeout:
            self.log.info('No kernels for %d seconds; shutting down.', seconds_since_active)
            self.stop()

    def init_shutdown_no_activity(self) -> None:
        """Initialize a shutdown on no activity."""
        if self.shutdown_no_activity_timeout > 0:
            self.log.info('Will shut down after %d seconds with no kernels.', self.shutdown_no_activity_timeout)
            pc = ioloop.PeriodicCallback(self.shutdown_no_activity, 60000)
            pc.start()

    @property
    def http_server(self) -> httpserver.HTTPServer:
        """An instance of Tornado's HTTPServer class for the Server Web Application."""
        try:
            return self._http_server
        except AttributeError:
            msg = 'An HTTPServer instance has not been created for the Server Web Application. To create an HTTPServer for this application, call `.init_httpserver()`.'
            raise AttributeError(msg) from None

    def init_httpserver(self) -> None:
        """Creates an instance of a Tornado HTTPServer for the Server Web Application
        and sets the http_server attribute.
        """
        if not hasattr(self, 'web_app'):
            msg = 'A tornado web application has not be initialized. Try calling `.init_webapp()` first.'
            raise AttributeError(msg)
        self._http_server = httpserver.HTTPServer(self.web_app, ssl_options=self.ssl_options, xheaders=self.trust_xheaders, max_body_size=self.max_body_size, max_buffer_size=self.max_buffer_size)
        if not self.sock:
            self._find_http_port()
        self.io_loop.add_callback(self._bind_http_server)

    def _bind_http_server(self) -> None:
        """Bind our http server."""
        success = self._bind_http_server_unix() if self.sock else self._bind_http_server_tcp()
        if not success:
            self.log.critical(_i18n('ERROR: the Jupyter server could not be started because no available port could be found.'))
            self.exit(1)

    def _bind_http_server_unix(self) -> bool:
        """Bind an http server on unix."""
        if unix_socket_in_use(self.sock):
            self.log.warning(_i18n('The socket %s is already in use.') % self.sock)
            return False
        try:
            sock = bind_unix_socket(self.sock, mode=int(self.sock_mode.encode(), 8))
            self.http_server.add_socket(sock)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                self.log.warning(_i18n('The socket %s is already in use.') % self.sock)
                return False
            elif e.errno in (errno.EACCES, getattr(errno, 'WSAEACCES', errno.EACCES)):
                self.log.warning(_i18n('Permission to listen on sock %s denied') % self.sock)
                return False
            else:
                raise
        else:
            return True

    def _bind_http_server_tcp(self) -> bool:
        """Bind a tcp server."""
        self.http_server.listen(self.port, self.ip)
        return True

    def _find_http_port(self) -> None:
        """Find an available http port."""
        success = False
        port = self.port
        for port in random_ports(self.port, self.port_retries + 1):
            try:
                sockets = bind_sockets(port, self.ip)
                sockets[0].close()
            except OSError as e:
                if e.errno == errno.EADDRINUSE:
                    if self.port_retries:
                        self.log.info(_i18n('The port %i is already in use, trying another port.') % port)
                    else:
                        self.log.info(_i18n('The port %i is already in use.') % port)
                    continue
                if e.errno in (errno.EACCES, getattr(errno, 'WSAEACCES', errno.EACCES)):
                    self.log.warning(_i18n('Permission to listen on port %i denied.') % port)
                    continue
                raise
            else:
                success = True
                self.port = port
                break
        if not success:
            if self.port_retries:
                self.log.critical(_i18n('ERROR: the Jupyter server could not be started because no available port could be found.'))
            else:
                self.log.critical(_i18n('ERROR: the Jupyter server could not be started because port %i is not available.') % port)
            self.exit(1)

    @staticmethod
    def _init_asyncio_patch() -> None:
        """set default asyncio policy to be compatible with tornado

        Tornado 6.0 is not compatible with default asyncio
        ProactorEventLoop, which lacks basic *_reader methods.
        Tornado 6.1 adds a workaround to add these methods in a thread,
        but SelectorEventLoop should still be preferred
        to avoid the extra thread for ~all of our events,
        at least until asyncio adds *_reader methods
        to proactor.
        """
        if sys.platform.startswith('win') and sys.version_info >= (3, 8):
            import asyncio
            try:
                from asyncio import WindowsProactorEventLoopPolicy, WindowsSelectorEventLoopPolicy
            except ImportError:
                pass
            else:
                if type(asyncio.get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
                    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

    @catch_config_error
    def initialize(self, argv: t.Optional[list[str]]=None, find_extensions: bool=True, new_httpserver: bool=True, starter_extension: t.Any=None) -> None:
        """Initialize the Server application class, configurables, web application, and http server.

        Parameters
        ----------
        argv : list or None
            CLI arguments to parse.
        find_extensions : bool
            If True, find and load extensions listed in Jupyter config paths. If False,
            only load extensions that are passed to ServerApp directly through
            the `argv`, `config`, or `jpserver_extensions` arguments.
        new_httpserver : bool
            If True, a tornado HTTPServer instance will be created and configured for the Server Web
            Application. This will set the http_server attribute of this class.
        starter_extension : str
            If given, it references the name of an extension point that started the Server.
            We will try to load configuration from extension point
        """
        self._init_asyncio_patch()
        super().initialize(argv=argv)
        if self._dispatching:
            return
        self.init_ioloop()
        if find_extensions:
            self.find_server_extensions()
        self.init_logging()
        self.init_event_logger()
        self.init_server_extensions()
        if starter_extension:
            point = self.extension_manager.extension_points[starter_extension]
            if point.app:
                self._starter_app = point.app
            self.update_config(Config(point.config))
        self.init_resources()
        self.init_configurables()
        self.init_components()
        self.init_webapp()
        self.init_signal()
        self.load_server_extensions()
        self.init_mime_overrides()
        self.init_shutdown_no_activity()
        if new_httpserver:
            self.init_httpserver()

    async def cleanup_kernels(self) -> None:
        """Shutdown all kernels.

        The kernels will shutdown themselves when this process no longer exists,
        but explicit shutdown allows the KernelManagers to cleanup the connection files.
        """
        if not getattr(self, 'kernel_manager', None):
            return
        n_kernels = len(self.kernel_manager.list_kernel_ids())
        kernel_msg = trans.ngettext('Shutting down %d kernel', 'Shutting down %d kernels', n_kernels)
        self.log.info(kernel_msg % n_kernels)
        await ensure_async(self.kernel_manager.shutdown_all())

    async def cleanup_extensions(self) -> None:
        """Call shutdown hooks in all extensions."""
        if not getattr(self, 'extension_manager', None):
            return
        n_extensions = len(self.extension_manager.extension_apps)
        extension_msg = trans.ngettext('Shutting down %d extension', 'Shutting down %d extensions', n_extensions)
        self.log.info(extension_msg % n_extensions)
        await ensure_async(self.extension_manager.stop_all_extensions())

    def running_server_info(self, kernel_count: bool=True) -> str:
        """Return the current working directory and the server url information"""
        info = t.cast(str, self.contents_manager.info_string()) + '\n'
        if kernel_count:
            n_kernels = len(self.kernel_manager.list_kernel_ids())
            kernel_msg = trans.ngettext('%d active kernel', '%d active kernels', n_kernels)
            info += kernel_msg % n_kernels
            info += '\n'
        info += _i18n(f'Jupyter Server {ServerApp.version} is running at:\n{self.display_url}')
        if self.gateway_config.gateway_enabled:
            info += _i18n('\nKernels will be managed by the Gateway server running at:\n%s') % self.gateway_config.url
        return info

    def server_info(self) -> dict[str, t.Any]:
        """Return a JSONable dict of information about this server."""
        return {'url': self.connection_url, 'hostname': self.ip if self.ip else 'localhost', 'port': self.port, 'sock': self.sock, 'secure': bool(self.certfile), 'base_url': self.base_url, 'token': self.identity_provider.token, 'root_dir': os.path.abspath(self.root_dir), 'password': bool(self.password), 'pid': os.getpid(), 'version': ServerApp.version}

    def write_server_info_file(self) -> None:
        """Write the result of server_info() to the JSON file info_file."""
        try:
            with secure_write(self.info_file) as f:
                json.dump(self.server_info(), f, indent=2, sort_keys=True)
        except OSError as e:
            self.log.error(_i18n('Failed to write server-info to %s: %r'), self.info_file, e)

    def remove_server_info_file(self) -> None:
        """Remove the jpserver-<pid>.json file created for this server.

        Ignores the error raised when the file has already been removed.
        """
        try:
            os.unlink(self.info_file)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def _resolve_file_to_run_and_root_dir(self) -> str:
        """Returns a relative path from file_to_run
        to root_dir. If root_dir and file_to_run
        are incompatible, i.e. on different subtrees,
        crash the app and log a critical message. Note
        that if root_dir is not configured and file_to_run
        is configured, root_dir will be set to the parent
        directory of file_to_run.
        """
        rootdir_abspath = pathlib.Path(self.root_dir).absolute()
        file_rawpath = pathlib.Path(self.file_to_run)
        combined_path = (rootdir_abspath / file_rawpath).absolute()
        is_child = str(combined_path).startswith(str(rootdir_abspath))
        if is_child:
            if combined_path.parent != rootdir_abspath:
                self.log.debug("The `root_dir` trait is set to a directory that's not the immediate parent directory of `file_to_run`. Note that the server will start at `root_dir` and open the the file from the relative path to the `root_dir`.")
            return str(combined_path.relative_to(rootdir_abspath))
        self.log.critical("`root_dir` and `file_to_run` are incompatible. They don't share the same subtrees. Make sure `file_to_run` is on the same path as `root_dir`.")
        self.exit(1)
        return ''

    def _write_browser_open_file(self, url: str, fh: t.Any) -> None:
        """Write the browser open file."""
        if self.identity_provider.token:
            url = url_concat(url, {'token': self.identity_provider.token})
        url = url_path_join(self.connection_url, url)
        jinja2_env = self.web_app.settings['jinja2_env']
        template = jinja2_env.get_template('browser-open.html')
        fh.write(template.render(open_url=url, base_url=self.base_url))

    def write_browser_open_files(self) -> None:
        """Write an `browser_open_file` and `browser_open_file_to_run` files

        This can be used to open a file directly in a browser.
        """
        self.write_browser_open_file()
        if self.file_to_run:
            file_to_run_relpath = self._resolve_file_to_run_and_root_dir()
            file_open_url = url_escape(url_path_join(self.file_url_prefix, *file_to_run_relpath.split(os.sep)))
            with open(self.browser_open_file_to_run, 'w', encoding='utf-8') as f:
                self._write_browser_open_file(file_open_url, f)

    def write_browser_open_file(self) -> None:
        """Write an jpserver-<pid>-open.html file

        This can be used to open the notebook in a browser
        """
        open_url = self.default_url[len(self.base_url):]
        with open(self.browser_open_file, 'w', encoding='utf-8') as f:
            self._write_browser_open_file(open_url, f)

    def remove_browser_open_files(self) -> None:
        """Remove the `browser_open_file` and `browser_open_file_to_run` files
        created for this server.

        Ignores the error raised when the file has already been removed.
        """
        self.remove_browser_open_file()
        try:
            os.unlink(self.browser_open_file_to_run)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def remove_browser_open_file(self) -> None:
        """Remove the jpserver-<pid>-open.html file created for this server.

        Ignores the error raised when the file has already been removed.
        """
        try:
            os.unlink(self.browser_open_file)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

    def _prepare_browser_open(self) -> tuple[str, t.Optional[str]]:
        """Prepare to open the browser."""
        if not self.use_redirect_file:
            uri = self.default_url[len(self.base_url):]
            if self.identity_provider.token:
                uri = url_concat(uri, {'token': self.identity_provider.token})
        if self.file_to_run:
            open_file = self.browser_open_file_to_run
        else:
            open_file = self.browser_open_file
        if self.use_redirect_file:
            assembled_url = urljoin('file:', pathname2url(open_file))
        else:
            assembled_url = url_path_join(self.connection_url, uri)
        return (assembled_url, open_file)

    def launch_browser(self) -> None:
        """Launch the browser."""
        import webbrowser
        try:
            browser = webbrowser.get(self.browser or None)
        except webbrowser.Error as e:
            self.log.warning(_i18n('No web browser found: %r.') % e)
            browser = None
        if not browser:
            return
        assembled_url, _ = self._prepare_browser_open()

        def target():
            assert browser is not None
            browser.open(assembled_url, new=self.webbrowser_open_new)
        threading.Thread(target=target).start()

    def start_app(self) -> None:
        """Start the Jupyter Server application."""
        super().start()
        if not self.allow_root:
            try:
                uid = os.geteuid()
            except AttributeError:
                uid = -1
            if uid == 0:
                self.log.critical(_i18n('Running as root is not recommended. Use --allow-root to bypass.'))
                self.exit(1)
        info = self.log.info
        for line in self.running_server_info(kernel_count=False).split('\n'):
            info(line)
        info(_i18n('Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).'))
        if 'dev' in __version__:
            info(_i18n('Welcome to Project Jupyter! Explore the various tools available and their corresponding documentation. If you are interested in contributing to the platform, please visit the community resources section at https://jupyter.org/community.html.'))
        self.write_server_info_file()
        if not self.no_browser_open_file:
            self.write_browser_open_files()
        if self.open_browser and (not self.sock):
            self.launch_browser()
        if self.identity_provider.token and self.identity_provider.token_generated:
            if self.sock:
                self.log.critical('\n'.join(['\n', 'Jupyter Server is listening on %s' % self.display_url, '', f'UNIX sockets are not browser-connectable, but you can tunnel to the instance via e.g.`ssh -L 8888:{self.sock} -N user@this_host` and then open e.g. {self.connection_url} in a browser.']))
            else:
                if self.no_browser_open_file:
                    message = ['\n', _i18n('To access the server, copy and paste one of these URLs:'), '    %s' % self.display_url]
                else:
                    message = ['\n', _i18n('To access the server, open this file in a browser:'), '    %s' % urljoin('file:', pathname2url(self.browser_open_file)), _i18n('Or copy and paste one of these URLs:'), '    %s' % self.display_url]
                self.log.critical('\n'.join(message))

    async def _cleanup(self) -> None:
        """General cleanup of files, extensions and kernels created
        by this instance ServerApp.
        """
        self.remove_server_info_file()
        self.remove_browser_open_files()
        await self.cleanup_extensions()
        await self.cleanup_kernels()
        try:
            await self.kernel_websocket_connection_class.close_all()
        except AttributeError:
            pass
        if getattr(self, 'kernel_manager', None):
            self.kernel_manager.__del__()
        if getattr(self, 'session_manager', None):
            self.session_manager.close()
        if hasattr(self, 'http_server'):
            self.http_server.stop()

    def start_ioloop(self) -> None:
        """Start the IO Loop."""
        if sys.platform.startswith('win'):
            pc = ioloop.PeriodicCallback(lambda: None, 5000)
            pc.start()
        try:
            self.io_loop.start()
        except KeyboardInterrupt:
            self.log.info(_i18n('Interrupted...'))

    def init_ioloop(self) -> None:
        """init self.io_loop so that an extension can use it by io_loop.call_later() to create background tasks"""
        self.io_loop = ioloop.IOLoop.current()

    def start(self) -> None:
        """Start the Jupyter server app, after initialization

        This method takes no arguments so all configuration and initialization
        must be done prior to calling this method."""
        self.start_app()
        self.start_ioloop()

    async def _stop(self) -> None:
        """Cleanup resources and stop the IO Loop."""
        await self._cleanup()
        if getattr(self, 'io_loop', None):
            self.io_loop.stop()

    def stop(self, from_signal: bool=False) -> None:
        """Cleanup resources and stop the server."""
        self._stopping = True
        if hasattr(self, 'http_server'):
            self.http_server.stop()
        if getattr(self, 'io_loop', None):
            if from_signal:
                self.io_loop.add_callback_from_signal(self._stop)
            else:
                self.io_loop.add_callback(self._stop)