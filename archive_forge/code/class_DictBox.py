import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
class DictBox(Box):
    __slots__ = []
    __getitem__ = container_take

    def __len__(self):
        return len(self._value)

    def __iter__(self):
        return self._value.__iter__()

    def __contains__(self, elt):
        return elt in self._value

    def items(self):
        return list(self.iteritems())

    def keys(self):
        return list(self.iterkeys())

    def values(self):
        return list(self.itervalues())

    def iteritems(self):
        return ((k, self[k]) for k in self)

    def iterkeys(self):
        return iter(self)

    def itervalues(self):
        return (self[k] for k in self)

    def get(self, k, d=None):
        return self[k] if k in self else d