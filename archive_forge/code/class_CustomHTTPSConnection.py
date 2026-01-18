from __future__ import (absolute_import, division, print_function)
import atexit
import base64
import email.mime.multipart
import email.mime.nonmultipart
import email.mime.application
import email.parser
import email.utils
import functools
import io
import mimetypes
import netrc
import os
import platform
import re
import socket
import sys
import tempfile
import traceback
import types  # pylint: disable=unused-import
from contextlib import contextmanager
import ansible.module_utils.compat.typing as t
import ansible.module_utils.six.moves.http_cookiejar as cookiejar
import ansible.module_utils.six.moves.urllib.error as urllib_error
from ansible.module_utils.common.collections import Mapping, is_sequence
from ansible.module_utils.six import PY2, PY3, string_types
from ansible.module_utils.six.moves import cStringIO
from ansible.module_utils.basic import get_distribution, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
class CustomHTTPSConnection(httplib.HTTPSConnection):

    def __init__(self, client_cert=None, client_key=None, *args, **kwargs):
        httplib.HTTPSConnection.__init__(self, *args, **kwargs)
        self.context = None
        if HAS_SSLCONTEXT:
            self.context = self._context
        elif HAS_URLLIB3_PYOPENSSLCONTEXT:
            self.context = self._context = PyOpenSSLContext(PROTOCOL)
        self._client_cert = client_cert
        self._client_key = client_key
        if self.context and self._client_cert:
            self.context.load_cert_chain(self._client_cert, self._client_key)

    def connect(self):
        """Connect to a host on a given (SSL) port."""
        if hasattr(self, 'source_address'):
            sock = socket.create_connection((self.host, self.port), self.timeout, self.source_address)
        else:
            sock = socket.create_connection((self.host, self.port), self.timeout)
        server_hostname = self.host
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
            server_hostname = self._tunnel_host
        if HAS_SSLCONTEXT or HAS_URLLIB3_PYOPENSSLCONTEXT:
            self.sock = self.context.wrap_socket(sock, server_hostname=server_hostname)
        elif HAS_URLLIB3_SSL_WRAP_SOCKET:
            self.sock = ssl_wrap_socket(sock, keyfile=self._client_key, cert_reqs=ssl.CERT_NONE, certfile=self._client_cert, ssl_version=PROTOCOL, server_hostname=server_hostname)
        else:
            self.sock = ssl.wrap_socket(sock, keyfile=self._client_key, certfile=self._client_cert, ssl_version=PROTOCOL)