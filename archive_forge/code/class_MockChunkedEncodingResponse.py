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
class MockChunkedEncodingResponse(object):

    def __init__(self, content):
        """
        content: collection of str, each str is a chunk in response
        """
        self.content = content
        self.index = 0
        self.closed = False
        self.cur_chunk = b''
        self.chunks_exhausted = False

    @staticmethod
    def _encode_chunk(chunk):
        length = '%X\r\n' % len(chunk)
        return length.encode() + chunk + b'\r\n'

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

    def pop_current_chunk(self, amt=-1, till_crlf=False):
        if amt > 0 and till_crlf:
            raise ValueError("Can't specify amt and till_crlf.")
        if len(self.cur_chunk) <= 0:
            self.cur_chunk = self._pop_new_chunk()
        if till_crlf:
            try:
                i = self.cur_chunk.index(b'\r\n')
            except ValueError:
                self.cur_chunk = b''
                return b''
            else:
                chunk_part = self.cur_chunk[:i + 2]
                self.cur_chunk = self.cur_chunk[i + 2:]
                return chunk_part
        elif amt <= -1:
            chunk_part = self.cur_chunk
            self.cur_chunk = b''
            return chunk_part
        else:
            try:
                chunk_part = self.cur_chunk[:amt]
            except IndexError:
                chunk_part = self.cur_chunk
                self.cur_chunk = b''
            else:
                self.cur_chunk = self.cur_chunk[amt:]
            return chunk_part

    def readline(self):
        return self.pop_current_chunk(till_crlf=True)

    def read(self, amt=-1):
        return self.pop_current_chunk(amt)

    def flush(self):
        pass

    def close(self):
        self.closed = True