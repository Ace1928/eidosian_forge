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
def _number_of_objects(self):
    """
        Return the number of shared objects
        """
    conn = self._Client(self._address, authkey=self._authkey)
    try:
        return dispatch(conn, None, 'number_of_objects')
    finally:
        conn.close()