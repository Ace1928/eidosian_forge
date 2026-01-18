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
def _send_stream(stream, write_func):
    write_func(b'chunked\n')
    _send_chunks(stream, write_func)
    write_func(b'END\n')