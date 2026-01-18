import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
def _validateChecksum(sentence):
    """
    Validates the checksum of an NMEA sentence.

    @param sentence: The NMEA sentence to check the checksum of.
    @type sentence: C{bytes}

    @raise ValueError: If the sentence has an invalid checksum.

    Simply returns on sentences that either don't have a checksum,
    or have a valid checksum.
    """
    if sentence[-3:-2] == b'*':
        reference, source = (int(sentence[-2:], 16), sentence[1:-3])
        computed = reduce(operator.xor, [ord(x) for x in iterbytes(source)])
        if computed != reference:
            raise base.InvalidChecksum(f'{computed:02x} != {reference:02x}')