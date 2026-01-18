import base64
import binascii
import datetime
import email.utils
import functools
import gzip
import hashlib
import hmac
import http.cookies
from inspect import isclass
from io import BytesIO
import mimetypes
import numbers
import os.path
import re
import socket
import sys
import threading
import time
import warnings
import tornado
import traceback
import types
import urllib.parse
from urllib.parse import urlencode
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import escape
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado import httputil
from tornado import iostream
from tornado import locale
from tornado.log import access_log, app_log, gen_log
from tornado import template
from tornado.escape import utf8, _unicode
from tornado.routing import (
from tornado.util import ObjectDict, unicode_type, _websocket_mask
from typing import (
from types import TracebackType
import typing
class FallbackHandler(RequestHandler):
    """A `RequestHandler` that wraps another HTTP server callback.

    The fallback is a callable object that accepts an
    `~.httputil.HTTPServerRequest`, such as an `Application` or
    `tornado.wsgi.WSGIContainer`.  This is most useful to use both
    Tornado ``RequestHandlers`` and WSGI in the same server.  Typical
    usage::

        wsgi_app = tornado.wsgi.WSGIContainer(
            django.core.handlers.wsgi.WSGIHandler())
        application = tornado.web.Application([
            (r"/foo", FooHandler),
            (r".*", FallbackHandler, dict(fallback=wsgi_app)),
        ])
    """

    def initialize(self, fallback: Callable[[httputil.HTTPServerRequest], None]) -> None:
        self.fallback = fallback

    def prepare(self) -> None:
        self.fallback(self.request)
        self._finished = True
        self.on_finish()