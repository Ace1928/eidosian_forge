import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
class ConstraintsExclusionTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.c1 = constraint.ConstraintsExclusion(constraint.ValueRangeConstraint(2, 4))

    def testGoodVal(self):
        try:
            self.c1(6)
        except error.ValueConstraintError:
            assert 0, 'constraint check fails'

    def testBadVal(self):
        try:
            self.c1(2)
        except error.ValueConstraintError:
            pass
        else:
            assert 0, 'constraint check fails'