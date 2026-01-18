import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
class ConstraintsIntersectionTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.c1 = constraint.ConstraintsIntersection(constraint.SingleValueConstraint(4), constraint.ValueRangeConstraint(2, 4))

    def testCmp1(self):
        assert constraint.SingleValueConstraint(4) in self.c1, '__cmp__() fails'

    def testCmp2(self):
        assert constraint.SingleValueConstraint(5) not in self.c1, '__cmp__() fails'

    def testCmp3(self):
        c = constraint.ConstraintsUnion(constraint.ConstraintsIntersection(constraint.SingleValueConstraint(4), constraint.ValueRangeConstraint(2, 4)))
        assert self.c1 in c, '__cmp__() fails'

    def testCmp4(self):
        c = constraint.ConstraintsUnion(constraint.ConstraintsIntersection(constraint.SingleValueConstraint(5)))
        assert self.c1 not in c, '__cmp__() fails'

    def testGoodVal(self):
        try:
            self.c1(4)
        except error.ValueConstraintError:
            assert 0, 'constraint check fails'

    def testBadVal(self):
        try:
            self.c1(-5)
        except error.ValueConstraintError:
            pass
        else:
            assert 0, 'constraint check fails'