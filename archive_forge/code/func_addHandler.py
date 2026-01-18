import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def addHandler(self, hdlr):
    """
        Add the specified handler to this logger.
        """
    _acquireLock()
    try:
        if not hdlr in self.handlers:
            self.handlers.append(hdlr)
    finally:
        _releaseLock()