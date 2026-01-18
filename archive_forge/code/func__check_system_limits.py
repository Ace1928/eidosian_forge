import os
from concurrent.futures import _base
import queue
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
from traceback import format_exception
def _check_system_limits():
    global _system_limits_checked, _system_limited
    if _system_limits_checked:
        if _system_limited:
            raise NotImplementedError(_system_limited)
    _system_limits_checked = True
    try:
        import multiprocessing.synchronize
    except ImportError:
        _system_limited = 'This Python build lacks multiprocessing.synchronize, usually due to named semaphores being unavailable on this platform.'
        raise NotImplementedError(_system_limited)
    try:
        nsems_max = os.sysconf('SC_SEM_NSEMS_MAX')
    except (AttributeError, ValueError):
        return
    if nsems_max == -1:
        return
    if nsems_max >= 256:
        return
    _system_limited = 'system provides too few semaphores (%d available, 256 necessary)' % nsems_max
    raise NotImplementedError(_system_limited)