import math
import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import error
from pyasn1.compat.octets import str2octs, ints2octs, octs2ints
from pyasn1.error import PyAsn1Error
class OctetStringWithUtf32TestCase(OctetStringWithUnicodeMixIn, BaseTestCase):
    initializer = (0, 0, 4, 48, 0, 0, 4, 49, 0, 0, 4, 50)
    encoding = 'utf-32-be'