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
class HTTP1ConnectionParameters(object):
    """Parameters for `.HTTP1Connection` and `.HTTP1ServerConnection`."""

    def __init__(self, no_keep_alive: bool=False, chunk_size: Optional[int]=None, max_header_size: Optional[int]=None, header_timeout: Optional[float]=None, max_body_size: Optional[int]=None, body_timeout: Optional[float]=None, decompress: bool=False) -> None:
        """
        :arg bool no_keep_alive: If true, always close the connection after
            one request.
        :arg int chunk_size: how much data to read into memory at once
        :arg int max_header_size:  maximum amount of data for HTTP headers
        :arg float header_timeout: how long to wait for all headers (seconds)
        :arg int max_body_size: maximum amount of data for body
        :arg float body_timeout: how long to wait while reading body (seconds)
        :arg bool decompress: if true, decode incoming
            ``Content-Encoding: gzip``
        """
        self.no_keep_alive = no_keep_alive
        self.chunk_size = chunk_size or 65536
        self.max_header_size = max_header_size or 65536
        self.header_timeout = header_timeout
        self.max_body_size = max_body_size
        self.body_timeout = body_timeout
        self.decompress = decompress