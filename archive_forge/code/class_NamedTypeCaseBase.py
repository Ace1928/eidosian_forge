import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
class NamedTypeCaseBase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.e = namedtype.NamedType('age', univ.Integer(0))

    def testIter(self):
        n, t = self.e
        assert n == 'age' or t == univ.Integer(), 'unpack fails'

    def testRepr(self):
        assert 'age' in repr(self.e)