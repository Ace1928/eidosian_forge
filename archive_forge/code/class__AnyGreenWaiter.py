import collections
import contextlib
import functools
from concurrent import futures
from concurrent.futures import _base
import futurist
from futurist import _utils
class _AnyGreenWaiter(object):
    """Provides the event that ``_wait_for_any_green`` blocks on."""

    def __init__(self):
        self.event = greenthreading.Event()

    def add_result(self, future):
        self.event.set()

    def add_exception(self, future):
        self.event.set()

    def add_cancelled(self, future):
        self.event.set()