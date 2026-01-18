import _collections_abc
import sys as _sys
from itertools import chain as _chain
from itertools import repeat as _repeat
from itertools import starmap as _starmap
from keyword import iskeyword as _iskeyword
from operator import eq as _eq
from operator import itemgetter as _itemgetter
from reprlib import recursive_repr as _recursive_repr
from _weakref import proxy as _proxy
class OrderedDict(dict):
    """Dictionary that remembers insertion order"""

    def __new__(cls, /, *args, **kwds):
        """Create the ordered dict object and set up the underlying structures."""
        self = dict.__new__(cls)
        self.__hardroot = _Link()
        self.__root = root = _proxy(self.__hardroot)
        root.prev = root.next = root
        self.__map = {}
        return self

    def __init__(self, other=(), /, **kwds):
        """Initialize an ordered dictionary.  The signature is the same as
        regular dictionaries.  Keyword argument order is preserved.
        """
        self.__update(other, **kwds)

    def __setitem__(self, key, value, dict_setitem=dict.__setitem__, proxy=_proxy, Link=_Link):
        """od.__setitem__(i, y) <==> od[i]=y"""
        if key not in self:
            self.__map[key] = link = Link()
            root = self.__root
            last = root.prev
            link.prev, link.next, link.key = (last, root, key)
            last.next = link
            root.prev = proxy(link)
        dict_setitem(self, key, value)

    def __delitem__(self, key, dict_delitem=dict.__delitem__):
        """od.__delitem__(y) <==> del od[y]"""
        dict_delitem(self, key)
        link = self.__map.pop(key)
        link_prev = link.prev
        link_next = link.next
        link_prev.next = link_next
        link_next.prev = link_prev
        link.prev = None
        link.next = None

    def __iter__(self):
        """od.__iter__() <==> iter(od)"""
        root = self.__root
        curr = root.next
        while curr is not root:
            yield curr.key
            curr = curr.next

    def __reversed__(self):
        """od.__reversed__() <==> reversed(od)"""
        root = self.__root
        curr = root.prev
        while curr is not root:
            yield curr.key
            curr = curr.prev

    def clear(self):
        """od.clear() -> None.  Remove all items from od."""
        root = self.__root
        root.prev = root.next = root
        self.__map.clear()
        dict.clear(self)

    def popitem(self, last=True):
        """Remove and return a (key, value) pair from the dictionary.

        Pairs are returned in LIFO order if last is true or FIFO order if false.
        """
        if not self:
            raise KeyError('dictionary is empty')
        root = self.__root
        if last:
            link = root.prev
            link_prev = link.prev
            link_prev.next = root
            root.prev = link_prev
        else:
            link = root.next
            link_next = link.next
            root.next = link_next
            link_next.prev = root
        key = link.key
        del self.__map[key]
        value = dict.pop(self, key)
        return (key, value)

    def move_to_end(self, key, last=True):
        """Move an existing element to the end (or beginning if last is false).

        Raise KeyError if the element does not exist.
        """
        link = self.__map[key]
        link_prev = link.prev
        link_next = link.next
        soft_link = link_next.prev
        link_prev.next = link_next
        link_next.prev = link_prev
        root = self.__root
        if last:
            last = root.prev
            link.prev = last
            link.next = root
            root.prev = soft_link
            last.next = link
        else:
            first = root.next
            link.prev = root
            link.next = first
            first.prev = soft_link
            root.next = link

    def __sizeof__(self):
        sizeof = _sys.getsizeof
        n = len(self) + 1
        size = sizeof(self.__dict__)
        size += sizeof(self.__map) * 2
        size += sizeof(self.__hardroot) * n
        size += sizeof(self.__root) * n
        return size
    update = __update = _collections_abc.MutableMapping.update

    def keys(self):
        """D.keys() -> a set-like object providing a view on D's keys"""
        return _OrderedDictKeysView(self)

    def items(self):
        """D.items() -> a set-like object providing a view on D's items"""
        return _OrderedDictItemsView(self)

    def values(self):
        """D.values() -> an object providing a view on D's values"""
        return _OrderedDictValuesView(self)
    __ne__ = _collections_abc.MutableMapping.__ne__
    __marker = object()

    def pop(self, key, default=__marker):
        """od.pop(k[,d]) -> v, remove specified key and return the corresponding
        value.  If key is not found, d is returned if given, otherwise KeyError
        is raised.

        """
        marker = self.__marker
        result = dict.pop(self, key, marker)
        if result is not marker:
            link = self.__map.pop(key)
            link_prev = link.prev
            link_next = link.next
            link_prev.next = link_next
            link_next.prev = link_prev
            link.prev = None
            link.next = None
            return result
        if default is marker:
            raise KeyError(key)
        return default

    def setdefault(self, key, default=None):
        """Insert key with a value of default if key is not in the dictionary.

        Return the value for key if key is in the dictionary, else default.
        """
        if key in self:
            return self[key]
        self[key] = default
        return default

    @_recursive_repr()
    def __repr__(self):
        """od.__repr__() <==> repr(od)"""
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self.items()))

    def __reduce__(self):
        """Return state information for pickling"""
        state = self.__getstate__()
        if state:
            if isinstance(state, tuple):
                state, slots = state
            else:
                slots = {}
            state = state.copy()
            slots = slots.copy()
            for k in vars(OrderedDict()):
                state.pop(k, None)
                slots.pop(k, None)
            if slots:
                state = (state, slots)
            else:
                state = state or None
        return (self.__class__, (), state, None, iter(self.items()))

    def copy(self):
        """od.copy() -> a shallow copy of od"""
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        """Create a new ordered dictionary with keys from iterable and values set to value.
        """
        self = cls()
        for key in iterable:
            self[key] = value
        return self

    def __eq__(self, other):
        """od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive
        while comparison to a regular mapping is order-insensitive.

        """
        if isinstance(other, OrderedDict):
            return dict.__eq__(self, other) and all(map(_eq, self, other))
        return dict.__eq__(self, other)

    def __ior__(self, other):
        self.update(other)
        return self

    def __or__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        new = self.__class__(self)
        new.update(other)
        return new

    def __ror__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        new = self.__class__(other)
        new.update(self)
        return new