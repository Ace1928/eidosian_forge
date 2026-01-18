import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import char
from pyasn1.type import univ
from pyasn1.type import constraint
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class UTF8StringTestCase(AbstractStringTestCase, BaseTestCase):
    initializer = (209, 132, 208, 176)
    encoding = 'utf-8'
    asn1Type = char.UTF8String