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
def _create_compressors(self, side: str, agreed_parameters: Dict[str, Any], compression_options: Optional[Dict[str, Any]]=None) -> None:
    allowed_keys = set(['server_no_context_takeover', 'client_no_context_takeover', 'server_max_window_bits', 'client_max_window_bits'])
    for key in agreed_parameters:
        if key not in allowed_keys:
            raise ValueError('unsupported compression parameter %r' % key)
    other_side = 'client' if side == 'server' else 'server'
    self._compressor = _PerMessageDeflateCompressor(**self._get_compressor_options(side, agreed_parameters, compression_options))
    self._decompressor = _PerMessageDeflateDecompressor(max_message_size=self.params.max_message_size, **self._get_compressor_options(other_side, agreed_parameters, compression_options))