import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import decoder
from pyasn1.codec.ber import eoo
from pyasn1.compat.octets import ints2octs, str2octs, null
from pyasn1.error import PyAsn1Error
class EndOfOctetsTestCase(BaseTestCase):

    def testUnexpectedEoo(self):
        try:
            decoder.decode(ints2octs((0, 0)))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'end-of-contents octets accepted at top level'

    def testExpectedEoo(self):
        result, remainder = decoder.decode(ints2octs((0, 0)), allowEoo=True)
        assert eoo.endOfOctets.isSameTypeWith(result) and result == eoo.endOfOctets and (result is eoo.endOfOctets)
        assert remainder == null

    def testDefiniteNoEoo(self):
        try:
            decoder.decode(ints2octs((35, 2, 0, 0)))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'end-of-contents octets accepted inside definite-length encoding'

    def testIndefiniteEoo(self):
        result, remainder = decoder.decode(ints2octs((35, 128, 0, 0)))
        assert result == () and remainder == null, 'incorrect decoding of indefinite length end-of-octets'

    def testNoLongFormEoo(self):
        try:
            decoder.decode(ints2octs((35, 128, 0, 129, 0)))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'end-of-contents octets accepted with invalid long-form length'

    def testNoConstructedEoo(self):
        try:
            decoder.decode(ints2octs((35, 128, 32, 0)))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'end-of-contents octets accepted with invalid constructed encoding'

    def testNoEooData(self):
        try:
            decoder.decode(ints2octs((35, 128, 0, 1, 0)))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'end-of-contents octets accepted with unexpected data'