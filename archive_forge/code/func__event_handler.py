import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
def _event_handler(self, new):
    with self._lock:
        self.events.append(new)
    self.event_count_updated = True