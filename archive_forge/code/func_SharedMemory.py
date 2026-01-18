import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
def SharedMemory(self, size):
    """Returns a new SharedMemory instance with the specified size in
            bytes, to be tracked by the manager."""
    with self._Client(self._address, authkey=self._authkey) as conn:
        sms = shared_memory.SharedMemory(None, create=True, size=size)
        try:
            dispatch(conn, None, 'track_segment', (sms.name,))
        except BaseException as e:
            sms.unlink()
            raise e
    return sms