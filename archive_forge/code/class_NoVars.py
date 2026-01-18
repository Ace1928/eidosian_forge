import binascii
import warnings
from webob.compat import (
class NoVars(object):
    """
    Represents no variables; used when no variables
    are applicable.

    This is read-only
    """

    def __init__(self, reason=None):
        self.reason = reason or 'N/A'

    def __getitem__(self, key):
        raise KeyError('No key %r: %s' % (key, self.reason))

    def __setitem__(self, *args, **kw):
        raise KeyError('Cannot add variables: %s' % self.reason)
    add = __setitem__
    setdefault = __setitem__
    update = __setitem__

    def __delitem__(self, *args, **kw):
        raise KeyError('No keys to delete: %s' % self.reason)
    clear = __delitem__
    pop = __delitem__
    popitem = __delitem__

    def get(self, key, default=None):
        return default

    def getall(self, key):
        return []

    def getone(self, key):
        return self[key]

    def mixed(self):
        return {}
    dict_of_lists = mixed

    def __contains__(self, key):
        return False
    has_key = __contains__

    def copy(self):
        return self

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.reason)

    def __len__(self):
        return 0

    def iterkeys(self):
        return iter([])
    if PY2:

        def __cmp__(self, other):
            return cmp({}, other)

        def keys(self):
            return []
        items = keys
        values = keys
        itervalues = iterkeys
        iteritems = iterkeys
    else:
        keys = iterkeys
        items = iterkeys
        values = iterkeys
    __iter__ = iterkeys