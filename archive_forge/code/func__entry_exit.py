import sys
from time import monotonic
from greenlet import GreenletExit
from kombu.asynchronous import timer as _timer
from celery import signals
from . import base
def _entry_exit(self, g, entry):
    try:
        try:
            g.wait()
        except self.GreenletExit:
            entry.cancel()
            g.canceled = True
    finally:
        self._queue.discard(g)