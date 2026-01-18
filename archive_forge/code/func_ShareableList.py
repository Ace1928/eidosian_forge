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
def ShareableList(self, sequence):
    """Returns a new ShareableList instance populated with the values
            from the input sequence, to be tracked by the manager."""
    with self._Client(self._address, authkey=self._authkey) as conn:
        sl = shared_memory.ShareableList(sequence)
        try:
            dispatch(conn, None, 'track_segment', (sl.shm.name,))
        except BaseException as e:
            sl.shm.unlink()
            raise e
    return sl