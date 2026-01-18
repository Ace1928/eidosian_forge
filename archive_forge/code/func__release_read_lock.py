import collections
import contextlib
import functools
import threading
from typing import Optional
from fasteners import _utils
def _release_read_lock(self, me, raise_on_not_owned=True):
    with self._cond:
        try:
            me_instances = self._readers[me]
            if me_instances > 1:
                self._readers[me] = me_instances - 1
            else:
                self._readers.pop(me)
        except KeyError:
            if raise_on_not_owned:
                raise RuntimeError(f'Thread {me} does not own a read lock')
        self._cond.notify_all()