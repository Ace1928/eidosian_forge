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
def _get_compressor_options(self, side: str, agreed_parameters: Dict[str, Any], compression_options: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """Converts a websocket agreed_parameters set to keyword arguments
        for our compressor objects.
        """
    options = dict(persistent=side + '_no_context_takeover' not in agreed_parameters)
    wbits_header = agreed_parameters.get(side + '_max_window_bits', None)
    if wbits_header is None:
        options['max_wbits'] = zlib.MAX_WBITS
    else:
        options['max_wbits'] = int(wbits_header)
    options['compression_options'] = compression_options
    return options