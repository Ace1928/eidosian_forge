import sys
from past.utils import with_metaclass
class olddict(with_metaclass(BaseOldDict, _builtin_dict)):
    """
    A backport of the Python 3 dict object to Py2
    """
    iterkeys = _builtin_dict.keys
    viewkeys = _builtin_dict.keys

    def keys(self):
        return list(super(olddict, self).keys())
    itervalues = _builtin_dict.values
    viewvalues = _builtin_dict.values

    def values(self):
        return list(super(olddict, self).values())
    iteritems = _builtin_dict.items
    viewitems = _builtin_dict.items

    def items(self):
        return list(super(olddict, self).items())

    def has_key(self, k):
        """
        D.has_key(k) -> True if D has a key k, else False
        """
        return k in self

    def __native__(self):
        """
        Hook for the past.utils.native() function
        """
        return super(oldbytes, self)