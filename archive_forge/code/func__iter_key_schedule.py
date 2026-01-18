import struct
from passlib import exc
from passlib.utils.compat import join_byte_values, byte_elem_value, \
def _iter_key_schedule(ks_odd):
    """given 64-bit key, iterates over the 8 (even,odd) key schedule pairs"""
    for p_even, p_odd in PCXROT:
        ks_even = _permute(ks_odd, p_even)
        ks_odd = _permute(ks_even, p_odd)
        yield (ks_even & _KS_MASK, ks_odd & _KS_MASK)