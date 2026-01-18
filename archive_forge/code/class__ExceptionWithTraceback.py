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
class _ExceptionWithTraceback:

    def __init__(self, exc, tb):
        tb = ''.join(format_exception(type(exc), exc, tb))
        self.exc = exc
        self.exc.__traceback__ = None
        self.tb = '\n"""\n%s"""' % tb

    def __reduce__(self):
        return (_rebuild_exc, (self.exc, self.tb))