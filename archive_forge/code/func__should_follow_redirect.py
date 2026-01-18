from tornado.escape import _unicode
from tornado import gen, version
from tornado.httpclient import (
from tornado import httputil
from tornado.http1connection import HTTP1Connection, HTTP1ConnectionParameters
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError, IOStream
from tornado.netutil import (
from tornado.log import gen_log
from tornado.tcpclient import TCPClient
import base64
import collections
import copy
import functools
import re
import socket
import ssl
import sys
import time
from io import BytesIO
import urllib.parse
from typing import Dict, Any, Callable, Optional, Type, Union
from types import TracebackType
import typing
def _should_follow_redirect(self) -> bool:
    if self.request.follow_redirects:
        assert self.request.max_redirects is not None
        return self.code in (301, 302, 303, 307, 308) and self.request.max_redirects > 0 and (self.headers is not None) and (self.headers.get('Location') is not None)
    return False