from collections.abc import MutableMapping
from weakref import ref
class WeakIDDict(MutableMapping):
    """ A weak-key-value dictionary that uses the id() of the key for
    comparisons.
    """

    def __init__(self, dict=None):
        self.data = {}
        if dict is not None:
            self.update(dict)

    def __repr__(self):
        return f'<{self.__class__.__name__} at 0x{id(self):x}>'

    def __delitem__(self, key):
        del self.data[id(key)]

    def __getitem__(self, key):
        return self.data[id(key)][1]()

    def __setitem__(self, key, value):
        self.data[id(key)] = (ref(key, _remover(id(key), ref(self))), ref(value, _remover(id(key), ref(self))))

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        return id(key) in self.data

    def __iter__(self):
        for id_key in self.data:
            wr_key = self.data[id_key][0]
            key = wr_key()
            if key is not None:
                yield key