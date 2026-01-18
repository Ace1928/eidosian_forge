import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import to_bytes
from pyasn1.compat.octets import (int2oct, oct2int, ints2octs, null,
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
@staticmethod
def _dropFloatingPoint(m, encbase, e):
    ms, es = (1, 1)
    if m < 0:
        ms = -1
    if e < 0:
        es = -1
    m *= ms
    if encbase == 8:
        m *= 2 ** (abs(e) % 3 * es)
        e = abs(e) // 3 * es
    elif encbase == 16:
        m *= 2 ** (abs(e) % 4 * es)
        e = abs(e) // 4 * es
    while True:
        if int(m) != m:
            m *= encbase
            e -= 1
            continue
        break
    return (ms, int(m), encbase, e)