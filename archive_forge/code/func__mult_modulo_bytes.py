import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
@staticmethod
def _mult_modulo_bytes(term1, term2, modulus):
    if not isinstance(term1, IntegerGMP):
        term1 = IntegerGMP(term1)
    if not isinstance(term2, IntegerGMP):
        term2 = IntegerGMP(term2)
    if not isinstance(modulus, IntegerGMP):
        modulus = IntegerGMP(modulus)
    if modulus < 0:
        raise ValueError('Modulus must be positive')
    if modulus == 0:
        raise ZeroDivisionError('Modulus cannot be zero')
    if modulus & 1 == 0:
        raise ValueError('Odd modulus is required')
    numbers_len = len(modulus.to_bytes())
    result = (term1 * term2 % modulus).to_bytes(numbers_len)
    return result