import collections
import weakref
from tensorflow.python.util import object_identity
def _set(self, key, value):
    may_affect_upstream = self.attributes[key].mark_as(value)
    if may_affect_upstream or self.always_propagate:
        for node in self._parents:
            node.invalidate(key)