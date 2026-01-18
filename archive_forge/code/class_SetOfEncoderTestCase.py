import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class SetOfEncoderTestCase(BaseTestCase):

    def testEmpty(self):
        s = univ.SetOf()
        assert encoder.encode(s) == ints2octs((49, 128, 0, 0))

    def testDefMode1(self):
        s = univ.SetOf()
        s.append(univ.OctetString('a'))
        s.append(univ.OctetString('ab'))
        assert encoder.encode(s) == ints2octs((49, 128, 4, 1, 97, 4, 2, 97, 98, 0, 0))

    def testDefMode2(self):
        s = univ.SetOf()
        s.append(univ.OctetString('ab'))
        s.append(univ.OctetString('a'))
        assert encoder.encode(s) == ints2octs((49, 128, 4, 1, 97, 4, 2, 97, 98, 0, 0))

    def testDefMode3(self):
        s = univ.SetOf()
        s.append(univ.OctetString('b'))
        s.append(univ.OctetString('a'))
        assert encoder.encode(s) == ints2octs((49, 128, 4, 1, 97, 4, 1, 98, 0, 0))

    def testDefMode4(self):
        s = univ.SetOf()
        s.append(univ.OctetString('a'))
        s.append(univ.OctetString('b'))
        assert encoder.encode(s) == ints2octs((49, 128, 4, 1, 97, 4, 1, 98, 0, 0))