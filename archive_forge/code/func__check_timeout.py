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
def _check_timeout(t):
    return getattr(time, 'monotonic', time.time)() > t