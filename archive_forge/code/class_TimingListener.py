import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from numba.core import config
class TimingListener(Listener):
    """A listener that measures the total time spent between *START* and
    *END* events during the time this listener is active.
    """

    def __init__(self):
        self._depth = 0

    def on_start(self, event):
        if self._depth == 0:
            self._ts = timer()
        self._depth += 1

    def on_end(self, event):
        self._depth -= 1
        if self._depth == 0:
            last = getattr(self, '_duration', 0)
            self._duration = timer() - self._ts + last

    @property
    def done(self):
        """Returns a ``bool`` indicating whether a measurement has been made.

        When this returns ``False``, the matching event has never fired.
        If and only if this returns ``True``, ``.duration`` can be read without
        error.
        """
        return hasattr(self, '_duration')

    @property
    def duration(self):
        """Returns the measured duration.

        This may raise ``AttributeError``. Users can use ``.done`` to check
        that a measurement has been made.
        """
        return self._duration