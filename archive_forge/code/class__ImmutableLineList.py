from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
class _ImmutableLineList(list):
    """
    Some protection for our 'lines' list, which is assumed to be immutable in the cache.
    (Useful for detecting obvious bugs.)
    """

    def _error(self, *a, **kw):
        raise NotImplementedError('Attempt to modifiy an immutable list.')
    __setitem__ = _error
    append = _error
    clear = _error
    extend = _error
    insert = _error
    pop = _error
    remove = _error
    reverse = _error
    sort = _error