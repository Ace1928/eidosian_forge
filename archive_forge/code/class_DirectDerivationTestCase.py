import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
class DirectDerivationTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.c1 = constraint.SingleValueConstraint(5)
        self.c2 = constraint.ConstraintsUnion(self.c1, constraint.ValueRangeConstraint(1, 3))

    def testGoodVal(self):
        assert self.c1.isSuperTypeOf(self.c2), 'isSuperTypeOf failed'
        assert not self.c1.isSubTypeOf(self.c2), 'isSubTypeOf failed'

    def testBadVal(self):
        assert not self.c2.isSuperTypeOf(self.c1), 'isSuperTypeOf failed'
        assert self.c2.isSubTypeOf(self.c1), 'isSubTypeOf failed'