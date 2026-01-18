import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import char
from pyasn1.type import univ
from pyasn1.type import constraint
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class GeneralStringTestCase(AbstractStringTestCase, BaseTestCase):
    initializer = (169, 174)
    encoding = 'iso-8859-1'
    asn1Type = char.GeneralString