import array
import threading
import time
from paramiko.util import b
def _buffer_tobytes(self, limit=None):
    return self._buffer[:limit].tobytes()