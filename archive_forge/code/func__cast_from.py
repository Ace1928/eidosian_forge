import ctypes, ctypes.util, operator, sys
from . import model
@classmethod
def _cast_from(cls, source):
    if isinstance(source, float):
        pass
    elif isinstance(source, CTypesGenericPrimitive):
        if hasattr(source, '__float__'):
            source = float(source)
        else:
            source = int(source)
    else:
        source = _cast_source_to_int(source)
    source = ctype(source).value
    return cls(source)