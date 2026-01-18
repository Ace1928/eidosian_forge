import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
def _mk_bitmap(bits, _CODEBITS=_CODEBITS, _int=int):
    s = bits.translate(_BITS_TRANS)[::-1]
    return [_int(s[i - _CODEBITS:i], 2) for i in range(len(s), 0, -_CODEBITS)]