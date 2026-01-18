from __future__ import annotations
from typing import Any, Mapping
def _getstate_slots(self: Any) -> Mapping[Any, Any]:
    prefix = self.__class__.__name__
    ret = {}
    for name in self.__slots__:
        mangled_name = _mangle_name(name, prefix)
        if hasattr(self, mangled_name):
            ret[mangled_name] = getattr(self, mangled_name)
    return ret