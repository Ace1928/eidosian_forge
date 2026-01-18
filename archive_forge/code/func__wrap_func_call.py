import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
@staticmethod
def _wrap_func_call(func):
    """Protect function call and return error with traceback."""
    try:
        return func()
    except BaseException as e:
        return _ExceptionWithTraceback(e)