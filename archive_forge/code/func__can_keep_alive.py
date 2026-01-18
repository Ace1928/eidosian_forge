import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
def _can_keep_alive(self, start_line: httputil.RequestStartLine, headers: httputil.HTTPHeaders) -> bool:
    if self.params.no_keep_alive:
        return False
    connection_header = headers.get('Connection')
    if connection_header is not None:
        connection_header = connection_header.lower()
    if start_line.version == 'HTTP/1.1':
        return connection_header != 'close'
    elif 'Content-Length' in headers or headers.get('Transfer-Encoding', '').lower() == 'chunked' or getattr(start_line, 'method', None) in ('HEAD', 'GET'):
        return connection_header == 'keep-alive'
    return False