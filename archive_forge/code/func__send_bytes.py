import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
def _send_bytes(self, buf):
    ov, err = _winapi.WriteFile(self._handle, buf, overlapped=True)
    try:
        if err == _winapi.ERROR_IO_PENDING:
            waitres = _winapi.WaitForMultipleObjects([ov.event], False, INFINITE)
            assert waitres == WAIT_OBJECT_0
    except:
        ov.cancel()
        raise
    finally:
        nwritten, err = ov.GetOverlappedResult(True)
    assert err == 0
    assert nwritten == len(buf)