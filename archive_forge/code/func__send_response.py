import _thread
import struct
import sys
import time
from collections import deque
from io import BytesIO
from fastbencode import bdecode_as_tuple, bencode
import breezy
from ... import debug, errors, osutils
from ...trace import log_exception_quietly, mutter
from . import message, request
def _send_response(self, response):
    """Send a smart server response down the output stream."""
    if self._finished:
        raise AssertionError('response already sent')
    self._finished = True
    self._write_protocol_version()
    self._write_success_or_failure_prefix(response)
    self._write_func(_encode_tuple(response.args))
    if response.body is not None:
        if not isinstance(response.body, bytes):
            raise AssertionError('body must be bytes')
        if not response.body_stream is None:
            raise AssertionError('body_stream and body cannot both be set')
        data = self._encode_bulk_data(response.body)
        self._write_func(data)
    elif response.body_stream is not None:
        _send_stream(response.body_stream, self._write_func)