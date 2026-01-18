import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class GeneralizedTimeEncoderTestCase(BaseTestCase):

    def testLocalTimezone(self):
        try:
            assert encoder.encode(useful.GeneralizedTime('20150501120112.1+0200'))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'Local timezone tolerated'

    def testMissingTimezone(self):
        try:
            assert encoder.encode(useful.GeneralizedTime('20150501120112.1'))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'Missing timezone tolerated'

    def testDecimalCommaPoint(self):
        try:
            assert encoder.encode(useful.GeneralizedTime('20150501120112,1Z'))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'Decimal comma tolerated'

    def testWithSubseconds(self):
        assert encoder.encode(useful.GeneralizedTime('20170801120112.59Z')) == ints2octs((24, 18, 50, 48, 49, 55, 48, 56, 48, 49, 49, 50, 48, 49, 49, 50, 46, 53, 57, 90))

    def testWithSeconds(self):
        assert encoder.encode(useful.GeneralizedTime('20170801120112Z')) == ints2octs((24, 15, 50, 48, 49, 55, 48, 56, 48, 49, 49, 50, 48, 49, 49, 50, 90))

    def testWithMinutes(self):
        assert encoder.encode(useful.GeneralizedTime('201708011201Z')) == ints2octs((24, 13, 50, 48, 49, 55, 48, 56, 48, 49, 49, 50, 48, 49, 90))