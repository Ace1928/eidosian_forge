import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.compat.octets import str2octs
from pyasn1.error import PyAsn1Error
class TaggedAnyTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.taggedAny = univ.Any().subtype(implicitTag=tag.Tag(tag.tagClassPrivate, tag.tagFormatSimple, 20))

        class Sequence(univ.Sequence):
            componentType = namedtype.NamedTypes(namedtype.NamedType('id', univ.Integer()), namedtype.NamedType('blob', self.taggedAny))
        self.s = Sequence()

    def testTypeCheckOnAssignment(self):
        self.s.clear()
        self.s['blob'] = self.taggedAny.clone('xxx')
        try:
            self.s.setComponentByName('blob', univ.Integer(123))
        except PyAsn1Error:
            pass
        else:
            assert False, 'non-open type assignment tolerated'