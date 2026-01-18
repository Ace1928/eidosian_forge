import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
class ValueSizeConstraintTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.c1 = constraint.ValueSizeConstraint(1, 2)

    def testGoodVal(self):
        try:
            self.c1('a')
        except error.ValueConstraintError:
            assert 0, 'constraint check fails'

    def testBadVal(self):
        try:
            self.c1('abc')
        except error.ValueConstraintError:
            pass
        else:
            assert 0, 'constraint check fails'