from __future__ import annotations
import collections
import io
import json as _json
import logging
import re
import sys
import typing
import warnings
import zlib
from contextlib import contextmanager
from http.client import HTTPMessage as _HttplibHTTPMessage
from http.client import HTTPResponse as _HttplibHTTPResponse
from socket import timeout as SocketTimeout
from . import util
from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPConnection, HTTPException
from .exceptions import (
from .util.response import is_fp_closed, is_response_to_head
from .util.retry import Retry
class ContentDecoder:

    def decompress(self, data: bytes) -> bytes:
        raise NotImplementedError()

    def flush(self) -> bytes:
        raise NotImplementedError()