import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
@staticmethod
def __normalizeBase10(value):
    m, b, e = value
    while m and m % 10 == 0:
        m /= 10
        e += 1
    return (m, b, e)