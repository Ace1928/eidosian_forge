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
class HTTPSClientAuthHandler(urllib_request.HTTPSHandler):
    """Handles client authentication via cert/key

        This is a fairly lightweight extension on HTTPSHandler, and can be used
        in place of HTTPSHandler
        """

    def __init__(self, client_cert=None, client_key=None, unix_socket=None, **kwargs):
        urllib_request.HTTPSHandler.__init__(self, **kwargs)
        self.client_cert = client_cert
        self.client_key = client_key
        self._unix_socket = unix_socket

    def https_open(self, req):
        return self.do_open(self._build_https_connection, req)

    def _build_https_connection(self, host, **kwargs):
        try:
            kwargs['context'] = self._context
        except AttributeError:
            pass
        if self._unix_socket:
            return UnixHTTPSConnection(self._unix_socket)(host, **kwargs)
        if not HAS_SSLCONTEXT:
            return CustomHTTPSConnection(host, client_cert=self.client_cert, client_key=self.client_key, **kwargs)
        return httplib.HTTPSConnection(host, **kwargs)