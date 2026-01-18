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
def _write_args(self, args):
    self._write_protocol_version()
    bytes = _encode_tuple(args)
    self._request.accept_bytes(bytes)