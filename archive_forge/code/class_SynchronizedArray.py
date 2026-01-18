import ctypes
import weakref
from . import heap
from . import get_context
from .context import reduction, assert_spawning
class SynchronizedArray(SynchronizedBase):

    def __len__(self):
        return len(self._obj)

    def __getitem__(self, i):
        with self:
            return self._obj[i]

    def __setitem__(self, i, value):
        with self:
            self._obj[i] = value

    def __getslice__(self, start, stop):
        with self:
            return self._obj[start:stop]

    def __setslice__(self, start, stop, values):
        with self:
            self._obj[start:stop] = values