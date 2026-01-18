import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from numba.core import config
def _setup_chrome_trace_exit_handler():
    """Setup a RecordingListener and an exit handler to write the captured
    events to file.
    """
    listener = RecordingListener()
    register('numba:run_pass', listener)
    filename = config.CHROME_TRACE

    @atexit.register
    def _write_chrome_trace():
        evs = _prepare_chrome_trace_data(listener)
        with open(filename, 'w') as out:
            json.dump(evs, out)