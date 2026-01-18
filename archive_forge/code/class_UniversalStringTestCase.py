import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import char
from pyasn1.type import univ
from pyasn1.type import constraint
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class UniversalStringTestCase(AbstractStringTestCase, BaseTestCase):
    initializer = (0, 0, 4, 48, 0, 0, 4, 68)
    encoding = 'utf-32-be'
    asn1Type = char.UniversalString