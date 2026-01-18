import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
def _get_attached_context(self):
    return getattr(self._tls, 'attached_context', None)