import collections
import contextlib
import functools
import threading
from typing import Optional
from fasteners import _utils
def _acquire_write_lock(self, me):
    if self.is_reader():
        raise RuntimeError('Reader %s to writer privilege escalation not allowed' % me)
    with self._cond:
        self._pending_writers.append(me)
        while True:
            if len(self._readers) == 0 and self._writer is None:
                if self._pending_writers[0] == me:
                    self._writer = self._pending_writers.popleft()
                    self._writer_entries = 1
                    break
            self._cond.wait()