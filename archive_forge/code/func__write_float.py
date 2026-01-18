import struct
import builtins
import warnings
from collections import namedtuple
def _write_float(f, x):
    import math
    if x < 0:
        sign = 32768
        x = x * -1
    else:
        sign = 0
    if x == 0:
        expon = 0
        himant = 0
        lomant = 0
    else:
        fmant, expon = math.frexp(x)
        if expon > 16384 or fmant >= 1 or fmant != fmant:
            expon = sign | 32767
            himant = 0
            lomant = 0
        else:
            expon = expon + 16382
            if expon < 0:
                fmant = math.ldexp(fmant, expon)
                expon = 0
            expon = expon | sign
            fmant = math.ldexp(fmant, 32)
            fsmant = math.floor(fmant)
            himant = int(fsmant)
            fmant = math.ldexp(fmant - fsmant, 32)
            fsmant = math.floor(fmant)
            lomant = int(fsmant)
    _write_ushort(f, expon)
    _write_ulong(f, himant)
    _write_ulong(f, lomant)