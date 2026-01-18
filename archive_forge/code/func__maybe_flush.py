import os
import sys
import signal
import itertools
import logging
import threading
from _weakrefset import WeakSet
from multiprocessing import process as _mproc
def _maybe_flush(f):
    try:
        f.flush()
    except (AttributeError, EnvironmentError, NotImplementedError):
        pass