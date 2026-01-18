import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
class SingleValueConstraintTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.c1 = constraint.SingleValueConstraint(1, 2)
        self.c2 = constraint.SingleValueConstraint(3, 4)

    def testCmp(self):
        assert self.c1 == self.c1, 'comparation fails'

    def testHash(self):
        assert hash(self.c1) != hash(self.c2), 'hash() fails'

    def testGoodVal(self):
        try:
            self.c1(1)
        except error.ValueConstraintError:
            assert 0, 'constraint check fails'

    def testBadVal(self):
        try:
            self.c1(4)
        except error.ValueConstraintError:
            pass
        else:
            assert 0, 'constraint check fails'