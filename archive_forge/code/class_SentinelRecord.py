import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
class SentinelRecord(object):
    """ Sentinel record to separate groups of chained change event dispatches.

    """
    __slots__ = ()

    def __str__(self):
        return '\n'