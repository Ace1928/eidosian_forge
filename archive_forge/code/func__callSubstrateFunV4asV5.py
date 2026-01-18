import io
import os
import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.codec.streaming import asSeekableStream
from pyasn1.codec.streaming import isEndOfStream
from pyasn1.codec.streaming import peekIntoStream
from pyasn1.codec.streaming import readFromStream
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import from_bytes
from pyasn1.compat.octets import oct2int, octs2ints, ints2octs, null
from pyasn1.error import PyAsn1Error
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import tagmap
from pyasn1.type import univ
from pyasn1.type import useful
@staticmethod
def _callSubstrateFunV4asV5(substrateFunV4, asn1Object, substrate, length):
    substrate_bytes = substrate.read()
    if length == -1:
        length = len(substrate_bytes)
    value, nextSubstrate = substrateFunV4(asn1Object, substrate_bytes, length)
    nbytes = substrate.write(nextSubstrate)
    substrate.truncate()
    substrate.seek(-nbytes, os.SEEK_CUR)
    yield value