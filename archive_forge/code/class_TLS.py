import logging
import sys
import types
import threading
import inspect
from functools import wraps
from itertools import chain
from numba.core import config
class TLS(threading.local):
    """Use a subclass to properly initialize the TLS variables in all threads."""

    def __init__(self):
        self.tracing = False
        self.indent = 0