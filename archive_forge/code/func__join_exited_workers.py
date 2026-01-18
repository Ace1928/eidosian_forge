import collections
import itertools
import os
import queue
import threading
import time
import traceback
import types
import warnings
from . import util
from . import get_context, TimeoutError
from .connection import wait
@staticmethod
def _join_exited_workers(pool):
    """Cleanup after any worker processes which have exited due to reaching
        their specified lifetime.  Returns True if any workers were cleaned up.
        """
    cleaned = False
    for i in reversed(range(len(pool))):
        worker = pool[i]
        if worker.exitcode is not None:
            util.debug('cleaning up worker %d' % i)
            worker.join()
            cleaned = True
            del pool[i]
    return cleaned