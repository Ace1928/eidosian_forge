import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _typeof(self, cdecl, consider_function_as_funcptr=False):
    try:
        result = self._parsed_types[cdecl]
    except KeyError:
        with self._lock:
            result = self._typeof_locked(cdecl)
    btype, really_a_function_type = result
    if really_a_function_type and (not consider_function_as_funcptr):
        raise CDefError('the type %r is a function type, not a pointer-to-function type' % (cdecl,))
    return btype