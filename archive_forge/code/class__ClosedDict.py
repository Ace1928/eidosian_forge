from pickle import DEFAULT_PROTOCOL, Pickler, Unpickler
from io import BytesIO
import collections.abc
class _ClosedDict(collections.abc.MutableMapping):
    """Marker for a closed dict.  Access attempts raise a ValueError."""

    def closed(self, *args):
        raise ValueError('invalid operation on closed shelf')
    __iter__ = __len__ = __getitem__ = __setitem__ = __delitem__ = keys = closed

    def __repr__(self):
        return '<Closed Dictionary>'