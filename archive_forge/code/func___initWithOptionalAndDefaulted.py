import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def __initWithOptionalAndDefaulted(self):
    self.s.clear()
    self.s.setComponentByPosition(0, univ.Null(''))
    self.s.setComponentByPosition(1, univ.OctetString('quick brown'))
    self.s.setComponentByPosition(2, univ.Integer(1))