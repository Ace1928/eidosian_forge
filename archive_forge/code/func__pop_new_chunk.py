import contextlib
import re
import socket
import ssl
import zlib
from base64 import b64decode
from io import BufferedReader, BytesIO, TextIOWrapper
from test import onlyBrotlipy
import mock
import pytest
import six
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.response import HTTPResponse, brotli
from urllib3.util.response import is_fp_closed
from urllib3.util.retry import RequestHistory, Retry
def _pop_new_chunk(self):
    if self.chunks_exhausted:
        return b''
    try:
        chunk = self.content[self.index]
    except IndexError:
        chunk = b''
        self.chunks_exhausted = True
    else:
        self.index += 1
    chunk = self._encode_chunk(chunk)
    if not isinstance(chunk, bytes):
        chunk = chunk.encode()
    return chunk