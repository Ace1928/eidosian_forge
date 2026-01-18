import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class UTCTimeEncoderTestCase(BaseTestCase):

    def testFractionOfSecond(self):
        try:
            assert encoder.encode(useful.UTCTime('150501120112.10Z'))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'Decimal point tolerated'

    def testMissingTimezone(self):
        try:
            assert encoder.encode(useful.UTCTime('150501120112')) == ints2octs((23, 13, 49, 53, 48, 53, 48, 49, 49, 50, 48, 49, 49, 50, 90))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'Missing timezone tolerated'

    def testLocalTimezone(self):
        try:
            assert encoder.encode(useful.UTCTime('150501120112+0200'))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'Local timezone tolerated'

    def testWithSeconds(self):
        assert encoder.encode(useful.UTCTime('990801120112Z')) == ints2octs((23, 13, 57, 57, 48, 56, 48, 49, 49, 50, 48, 49, 49, 50, 90))

    def testWithMinutes(self):
        assert encoder.encode(useful.UTCTime('9908011201Z')) == ints2octs((23, 11, 57, 57, 48, 56, 48, 49, 49, 50, 48, 49, 90))