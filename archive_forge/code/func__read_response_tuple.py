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
def _read_response_tuple(self):
    result = self._recv_tuple()
    if 'hpss' in debug.debug_flags:
        if self._request_start_time is not None:
            mutter('   result:   %6.3fs  %s', osutils.perf_counter() - self._request_start_time, repr(result)[1:-1])
            self._request_start_time = None
        else:
            mutter('   result:   %s', repr(result)[1:-1])
    return result