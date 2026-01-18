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
def _write_success_or_failure_prefix(self, response):
    """Write the protocol specific success/failure prefix."""
    if response.is_successful():
        self._write_func(b'success\n')
    else:
        self._write_func(b'failed\n')