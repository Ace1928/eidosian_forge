import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def _get_greenlet_id():
    curr_greenlet = getcurrent()
    id_ = getattr(curr_greenlet, '_yappi_tid', None)
    if id_ is None:
        id_ = GREENLET_COUNTER()
        curr_greenlet._yappi_tid = id_
    return id_