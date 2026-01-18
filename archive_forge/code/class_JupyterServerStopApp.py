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
class JupyterServerStopApp(JupyterApp):
    """An application to stop a Jupyter server."""
    version: str = __version__
    description: str = 'Stop currently running Jupyter server for a given port'
    port = Integer(DEFAULT_JUPYTER_SERVER_PORT, config=True, help='Port of the server to be killed. Default %s' % DEFAULT_JUPYTER_SERVER_PORT)
    sock = Unicode('', config=True, help='UNIX socket of the server to be killed.')

    def parse_command_line(self, argv=None):
        """Parse command line options."""
        super().parse_command_line(argv)
        if self.extra_args:
            try:
                self.port = int(self.extra_args[0])
            except ValueError:
                self.sock = self.extra_args[0]

    def shutdown_server(self, server):
        """Shut down a server."""
        return shutdown_server(server, log=self.log)

    def _shutdown_or_exit(self, target_endpoint, server):
        """Handle a shutdown."""
        self.log.info('Shutting down server on %s...' % target_endpoint)
        if not self.shutdown_server(server):
            sys.exit('Could not stop server on %s' % target_endpoint)

    @staticmethod
    def _maybe_remove_unix_socket(socket_path):
        """Try to remove a socket path."""
        try:
            os.unlink(socket_path)
        except OSError:
            pass

    def start(self):
        """Start the server stop app."""
        info = self.log.info
        servers = list(list_running_servers(self.runtime_dir, log=self.log))
        if not servers:
            self.exit('There are no running servers (per %s)' % self.runtime_dir)
        for server in servers:
            if self.sock:
                sock = server.get('sock', None)
                if sock and sock == self.sock:
                    self._shutdown_or_exit(sock, server)
                    self._maybe_remove_unix_socket(sock)
                    return
            elif self.port:
                port = server.get('port', None)
                if port == self.port:
                    self._shutdown_or_exit(port, server)
                    return
        current_endpoint = self.sock or self.port
        info(f'There is currently no server running on {current_endpoint}')
        info('Ports/sockets currently in use:')
        for server in servers:
            info(' - {}'.format(server.get('sock') or server['port']))
        self.exit(1)