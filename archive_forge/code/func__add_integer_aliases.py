from numpy.compat import unicode
from numpy.core._string_helpers import english_lower
from numpy.core.multiarray import typeinfo, dtype
from numpy.core._dtype import _kind_name
def _add_integer_aliases():
    seen_bits = set()
    for i_ctype, u_ctype in zip(_int_ctypes, _uint_ctypes):
        i_info = _concrete_typeinfo[i_ctype]
        u_info = _concrete_typeinfo[u_ctype]
        bits = i_info.bits
        for info, charname, intname in [(i_info, 'i%d' % (bits // 8,), 'int%d' % bits), (u_info, 'u%d' % (bits // 8,), 'uint%d' % bits)]:
            if bits not in seen_bits:
                allTypes[intname] = info.type
                sctypeDict[intname] = info.type
                sctypeDict[charname] = info.type
        seen_bits.add(bits)