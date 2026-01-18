from numpy.compat import unicode
from numpy.core._string_helpers import english_lower
from numpy.core.multiarray import typeinfo, dtype
from numpy.core._dtype import _kind_name
def _bits_of(obj):
    try:
        info = next((v for v in _concrete_typeinfo.values() if v.type is obj))
    except StopIteration:
        if obj in _abstract_types.values():
            msg = 'Cannot count the bits of an abstract type'
            raise ValueError(msg) from None
        return dtype(obj).itemsize * 8
    else:
        return info.bits