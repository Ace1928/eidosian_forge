import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
class ConstraintsIntersectionRangeTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.c1 = constraint.ConstraintsIntersection(constraint.ValueRangeConstraint(1, 9), constraint.ValueRangeConstraint(2, 5))

    def testGoodVal(self):
        try:
            self.c1(3)
        except error.ValueConstraintError:
            assert 0, 'constraint check fails'

    def testBadVal(self):
        try:
            self.c1(0)
        except error.ValueConstraintError:
            pass
        else:
            assert 0, 'constraint check fails'