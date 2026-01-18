from collections.abc import MutableMapping
from weakref import ref
class WeakIDKeyDict(WeakIDDict):
    """ A weak-key dictionary that uses the id() of the key for comparisons.

    This differs from `WeakIDDict` in that it does not try to make a weakref to
    the values.
    """

    def __getitem__(self, key):
        return self.data[id(key)][1]

    def __setitem__(self, key, value):
        self.data[id(key)] = (ref(key, _remover(id(key), ref(self))), value)