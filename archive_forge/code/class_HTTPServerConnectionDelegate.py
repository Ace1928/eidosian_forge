import calendar
import collections.abc
import copy
import datetime
import email.utils
from functools import lru_cache
from http.client import responses
import http.cookies
import re
from ssl import SSLError
import time
import unicodedata
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from tornado.escape import native_str, parse_qs_bytes, utf8
from tornado.log import gen_log
from tornado.util import ObjectDict, unicode_type
import typing
from typing import (
class HTTPServerConnectionDelegate(object):
    """Implement this interface to handle requests from `.HTTPServer`.

    .. versionadded:: 4.0
    """

    def start_request(self, server_conn: object, request_conn: 'HTTPConnection') -> 'HTTPMessageDelegate':
        """This method is called by the server when a new request has started.

        :arg server_conn: is an opaque object representing the long-lived
            (e.g. tcp-level) connection.
        :arg request_conn: is a `.HTTPConnection` object for a single
            request/response exchange.

        This method should return a `.HTTPMessageDelegate`.
        """
        raise NotImplementedError()

    def on_close(self, server_conn: object) -> None:
        """This method is called when a connection has been closed.

        :arg server_conn: is a server connection that has previously been
            passed to ``start_request``.
        """
        pass