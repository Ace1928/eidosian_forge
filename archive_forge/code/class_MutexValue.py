import os
from threading import Lock
import warnings
from .mmap_dict import mmap_key, MmapedDict
class MutexValue:
    """A float protected by a mutex."""
    _multiprocess = False

    def __init__(self, typ, metric_name, name, labelnames, labelvalues, help_text, **kwargs):
        self._value = 0.0
        self._exemplar = None
        self._lock = Lock()

    def inc(self, amount):
        with self._lock:
            self._value += amount

    def set(self, value, timestamp=None):
        with self._lock:
            self._value = value

    def set_exemplar(self, exemplar):
        with self._lock:
            self._exemplar = exemplar

    def get(self):
        with self._lock:
            return self._value

    def get_exemplar(self):
        with self._lock:
            return self._exemplar