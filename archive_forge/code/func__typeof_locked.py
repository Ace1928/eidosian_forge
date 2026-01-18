import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def _typeof_locked(self, cdecl):
    key = cdecl
    if key in self._parsed_types:
        return self._parsed_types[key]
    if not isinstance(cdecl, str):
        cdecl = cdecl.encode('ascii')
    type = self._parser.parse_type(cdecl)
    really_a_function_type = type.is_raw_function
    if really_a_function_type:
        type = type.as_function_pointer()
    btype = self._get_cached_btype(type)
    result = (btype, really_a_function_type)
    self._parsed_types[key] = result
    return result