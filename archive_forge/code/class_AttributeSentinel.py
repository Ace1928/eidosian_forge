import collections
import weakref
from tensorflow.python.util import object_identity
class AttributeSentinel(object):
    """Container for managing attribute cache state within a Layer.

  The cache can be invalidated either on an individual basis (for instance when
  an attribute is mutated) or a layer-wide basis (such as when a new dependency
  is added).
  """

    def __init__(self, always_propagate=False):
        self._parents = weakref.WeakSet()
        self.attributes = collections.defaultdict(MutationSentinel)
        self.always_propagate = always_propagate

    def __repr__(self):
        return '{}\n  {}'.format(super(AttributeSentinel, self).__repr__(), {k: v.in_cached_state for k, v in self.attributes.items()})

    def add_parent(self, node):
        self._parents.add(node)
        node.invalidate_all()

    def get(self, key):
        return self.attributes[key].in_cached_state

    def _set(self, key, value):
        may_affect_upstream = self.attributes[key].mark_as(value)
        if may_affect_upstream or self.always_propagate:
            for node in self._parents:
                node.invalidate(key)

    def mark_cached(self, key):
        self._set(key, True)

    def invalidate(self, key):
        self._set(key, False)

    def invalidate_all(self):
        for key in self.attributes.keys():
            self.attributes[key].mark_as(False)
        for node in self._parents:
            node.invalidate_all()