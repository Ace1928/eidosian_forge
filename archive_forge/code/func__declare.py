from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _declare(self, name, obj, included=False, quals=0):
    if name in self._declarations:
        prevobj, prevquals = self._declarations[name]
        if prevobj is obj and prevquals == quals:
            return
        if not self._options.get('override'):
            raise FFIError('multiple declarations of %s (for interactive usage, try cdef(xx, override=True))' % (name,))
    assert '__dotdotdot__' not in name.split()
    self._declarations[name] = (obj, quals)
    if included:
        self._included_declarations.add(obj)