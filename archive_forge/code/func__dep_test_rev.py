from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def _dep_test_rev(self, *deps):

    def assertGreater(a, b):
        self.assertTrue(a > b, '"%s" is not greater than "%s"' % (str(a), str(b)))
    self._dep_test(reversed, assertGreater, deps)