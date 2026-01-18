import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
class ValueRangeConstraintTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.c1 = constraint.ValueRangeConstraint(1, 4)

    def testGoodVal(self):
        try:
            self.c1(1)
        except error.ValueConstraintError:
            assert 0, 'constraint check fails'

    def testBadVal(self):
        try:
            self.c1(-5)
        except error.ValueConstraintError:
            pass
        else:
            assert 0, 'constraint check fails'