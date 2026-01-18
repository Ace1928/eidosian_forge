import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
def _sig_str_to_tuple(self, sig):
    sig = sig.replace(' ', '')
    toks = sig.split('->')
    if len(toks) != 2:
        raise ValueError(f'signature {sig} for dtypes is invalid')
    else:
        ins, outs = toks
    return (ins, outs)