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
def _check_readable(self):
    if not self._readable:
        raise OSError('connection is write-only')