import abc
import asyncio
import base64
import hashlib
import os
import sys
import struct
import tornado
from urllib.parse import urlparse
import warnings
import zlib
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen, httpclient, httputil
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.iostream import StreamClosedError, IOStream
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado import simple_httpclient
from tornado.queues import Queue
from tornado.tcpclient import TCPClient
from tornado.util import _websocket_mask
from typing import (
from types import TracebackType
def _raise_not_supported_for_websockets(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError('Method not supported for Web Sockets')