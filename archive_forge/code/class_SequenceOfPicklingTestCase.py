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
class SequenceOfPicklingTestCase(unittest.TestCase):

    def testSchemaPickling(self):
        old_asn1 = univ.SequenceOf(componentType=univ.OctetString())
        serialised = pickle.dumps(old_asn1)
        assert serialised
        new_asn1 = pickle.loads(serialised)
        assert type(new_asn1) == univ.SequenceOf
        assert old_asn1.isSameTypeWith(new_asn1)

    def testValuePickling(self):
        old_asn1 = univ.SequenceOf(componentType=univ.OctetString())
        old_asn1[0] = 'test'
        serialised = pickle.dumps(old_asn1)
        assert serialised
        new_asn1 = pickle.loads(serialised)
        assert new_asn1
        assert new_asn1 == [str2octs('test')]