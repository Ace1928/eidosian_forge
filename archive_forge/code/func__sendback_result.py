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
def _sendback_result(result_queue, work_id, result=None, exception=None, exit_pid=None):
    """Safely send back the given result or exception"""
    try:
        result_queue.put(_ResultItem(work_id, result=result, exception=exception, exit_pid=exit_pid))
    except BaseException as e:
        exc = _ExceptionWithTraceback(e, e.__traceback__)
        result_queue.put(_ResultItem(work_id, exception=exc, exit_pid=exit_pid))