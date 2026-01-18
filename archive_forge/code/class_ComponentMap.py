import collections
from collections.abc import Mapping as collections_Mapping
from pyomo.common.autoslots import AutoSlots
class ComponentMap(AutoSlots.Mixin, collections.abc.MutableMapping):
    """
    This class is a replacement for dict that allows Pyomo
    modeling components to be used as entry keys. The
    underlying mapping is based on the Python id() of the
    object, which gets around the problem of hashing
    subclasses of NumericValue. This class is meant for
    creating mappings from Pyomo components to values. The
    use of non-Pyomo components as entry keys should be
    avoided.

    A reference to the object is kept around as long as it
    has a corresponding entry in the container, so there is
    no need to worry about id() clashes.

    We also override __setstate__ so that we can rebuild the
    container based on possibly updated object ids after
    a deepcopy or pickle.

    *** An instance of this class should never be
    deepcopied/pickled unless it is done so along with the
    components for which it contains map entries (e.g., as
    part of a block). ***
    """
    __slots__ = ('_dict',)
    __autoslot_mappers__ = {'_dict': _rehash_keys}

    def __init__(self, *args, **kwds):
        self._dict = {}
        self.update(*args, **kwds)

    def __str__(self):
        """String representation of the mapping."""
        tmp = {f'{v[0]} (key={k})': v[1] for k, v in self._dict.items()}
        return f'ComponentMap({tmp})'

    def __getitem__(self, obj):
        try:
            return self._dict[_hasher[obj.__class__](obj)][1]
        except KeyError:
            _id = _hasher[obj.__class__](obj)
            raise KeyError(f'{obj} (key={_id})') from None

    def __setitem__(self, obj, val):
        self._dict[_hasher[obj.__class__](obj)] = (obj, val)

    def __delitem__(self, obj):
        try:
            del self._dict[_hasher[obj.__class__](obj)]
        except KeyError:
            _id = _hasher[obj.__class__](obj)
            raise KeyError(f'{obj} (key={_id})') from None

    def __iter__(self):
        return (obj for obj, val in self._dict.values())

    def __len__(self):
        return self._dict.__len__()

    def update(self, *args, **kwargs):
        if len(args) == 1 and (not kwargs) and isinstance(args[0], ComponentMap):
            return self._dict.update(args[0]._dict)
        return super().update(*args, **kwargs)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, collections_Mapping) or len(self) != len(other):
            return False
        for key, val in other.items():
            other_id = _hasher[key.__class__](key)
            if other_id not in self._dict:
                return False
            self_val = self._dict[other_id][1]
            if self_val is not val and self_val != val:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __contains__(self, obj):
        return _hasher[obj.__class__](obj) in self._dict

    def clear(self):
        """D.clear() -> None.  Remove all items from D."""
        self._dict.clear()

    def get(self, key, default=None):
        """D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."""
        if key in self:
            return self[key]
        return default

    def setdefault(self, key, default=None):
        """D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D"""
        if key in self:
            return self[key]
        else:
            self[key] = default
        return default