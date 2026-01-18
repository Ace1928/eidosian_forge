import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def _bootstrap_inner(self):
    try:
        self._set_ident()
        self._set_tstate_lock()
        if _HAVE_THREAD_NATIVE_ID:
            self._set_native_id()
        self._started.set()
        with _active_limbo_lock:
            _active[self._ident] = self
            del _limbo[self]
        if _trace_hook:
            _sys.settrace(_trace_hook)
        if _profile_hook:
            _sys.setprofile(_profile_hook)
        try:
            self.run()
        except:
            self._invoke_excepthook(self)
    finally:
        self._delete()