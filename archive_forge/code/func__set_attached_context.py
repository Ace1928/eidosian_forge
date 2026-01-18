import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
def _set_attached_context(self, ctx):
    self._tls.attached_context = ctx