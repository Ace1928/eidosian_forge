import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class _Lock(object):
    """Wrapper class of threading.Lock(), which is allowed by 'with'."""

    def __new__(cls):
        self = object.__new__(cls)
        self._lock = threading.Lock()
        return self

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._lock.release()