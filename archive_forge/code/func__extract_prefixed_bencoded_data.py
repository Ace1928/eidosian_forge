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
def _extract_prefixed_bencoded_data(self):
    prefixed_bytes = self._extract_length_prefixed_bytes()
    try:
        decoded = bdecode_as_tuple(prefixed_bytes)
    except ValueError:
        raise errors.SmartProtocolError('Bytes {!r} not bencoded'.format(prefixed_bytes))
    return decoded