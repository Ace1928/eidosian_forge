import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class OPTNonStandardAttributes:
    """
    Generate byte and instance representations of an L{dns._OPTHeader}
    where all attributes are set to non-default values.

    For testing whether attributes have really been read from the byte
    string during decoding.
    """

    @classmethod
    def bytes(cls, excludeName=False, excludeOptions=False):
        """
        Return L{bytes} representing an encoded OPT record.

        @param excludeName: A flag that controls whether to exclude
            the name field. This allows a non-standard name to be
            prepended during the test.
        @type excludeName: L{bool}

        @param excludeOptions: A flag that controls whether to exclude
            the RDLEN field. This allows encoded variable options to be
            appended during the test.
        @type excludeOptions: L{bool}

        @return: L{bytes} representing the encoded OPT record returned
            by L{object}.
        """
        rdlen = b'\x00\x00'
        if excludeOptions:
            rdlen = b''
        return b'\x00\x00)\x02\x00\x03\x04\x80\x00' + rdlen

    @classmethod
    def object(cls):
        """
        Return a new L{dns._OPTHeader} instance.

        @return: A L{dns._OPTHeader} instance with attributes that
            match the encoded record returned by L{bytes}.
        """
        return dns._OPTHeader(udpPayloadSize=512, extendedRCODE=3, version=4, dnssecOK=True)