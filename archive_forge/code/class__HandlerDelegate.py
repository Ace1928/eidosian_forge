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
class _HandlerDelegate(httputil.HTTPMessageDelegate):

    def __init__(self, application: Application, request: httputil.HTTPServerRequest, handler_class: Type[RequestHandler], handler_kwargs: Optional[Dict[str, Any]], path_args: Optional[List[bytes]], path_kwargs: Optional[Dict[str, bytes]]) -> None:
        self.application = application
        self.connection = request.connection
        self.request = request
        self.handler_class = handler_class
        self.handler_kwargs = handler_kwargs or {}
        self.path_args = path_args or []
        self.path_kwargs = path_kwargs or {}
        self.chunks = []
        self.stream_request_body = _has_stream_request_body(self.handler_class)

    def headers_received(self, start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], headers: httputil.HTTPHeaders) -> Optional[Awaitable[None]]:
        if self.stream_request_body:
            self.request._body_future = Future()
            return self.execute()
        return None

    def data_received(self, data: bytes) -> Optional[Awaitable[None]]:
        if self.stream_request_body:
            return self.handler.data_received(data)
        else:
            self.chunks.append(data)
            return None

    def finish(self) -> None:
        if self.stream_request_body:
            future_set_result_unless_cancelled(self.request._body_future, None)
        else:
            self.request.body = b''.join(self.chunks)
            self.request._parse_body()
            self.execute()

    def on_connection_close(self) -> None:
        if self.stream_request_body:
            self.handler.on_connection_close()
        else:
            self.chunks = None

    def execute(self) -> Optional[Awaitable[None]]:
        if not self.application.settings.get('compiled_template_cache', True):
            with RequestHandler._template_loader_lock:
                for loader in RequestHandler._template_loaders.values():
                    loader.reset()
        if not self.application.settings.get('static_hash_cache', True):
            static_handler_class = self.application.settings.get('static_handler_class', StaticFileHandler)
            static_handler_class.reset()
        self.handler = self.handler_class(self.application, self.request, **self.handler_kwargs)
        transforms = [t(self.request) for t in self.application.transforms]
        if self.stream_request_body:
            self.handler._prepared_future = Future()
        fut = gen.convert_yielded(self.handler._execute(transforms, *self.path_args, **self.path_kwargs))
        fut.add_done_callback(lambda f: f.result())
        return self.handler._prepared_future