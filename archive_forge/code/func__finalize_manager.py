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
@staticmethod
def _finalize_manager(process, address, authkey, state, _Client):
    """
        Shutdown the manager process; will be registered as a finalizer
        """
    if process.is_alive():
        util.info('sending shutdown message to manager')
        try:
            conn = _Client(address, authkey=authkey)
            try:
                dispatch(conn, None, 'shutdown')
            finally:
                conn.close()
        except Exception:
            pass
        process.join(timeout=1.0)
        if process.is_alive():
            util.info('manager still alive')
            if hasattr(process, 'terminate'):
                util.info('trying to `terminate()` manager process')
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    util.info('manager still alive after terminate')
    state.value = State.SHUTDOWN
    try:
        del BaseProxy._address_to_local[address]
    except KeyError:
        pass