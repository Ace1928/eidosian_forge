from __future__ import annotations
import ctypes
import ctypes.util
import sys
import traceback
from functools import partial
from itertools import count
from threading import Lock, Thread
from typing import Any, Callable, Generic, TypeVar
import outcome
def _handle_job(self) -> None:
    assert self._job is not None
    fn, deliver, name = self._job
    self._job = None
    if name is not None:
        self._thread.name = name
        if set_os_thread_name:
            set_os_thread_name(self._thread.ident, name)
    result = outcome.capture(fn)
    if name is not None:
        self._thread.name = self._default_name
        if set_os_thread_name:
            set_os_thread_name(self._thread.ident, self._default_name)
    self._thread_cache._idle_workers[self] = None
    try:
        deliver(result)
    except BaseException as e:
        print('Exception while delivering result of thread', file=sys.stderr)
        traceback.print_exception(type(e), e, e.__traceback__)