from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def _dep_test_fwd(self, *deps):

    def assertLess(a, b):
        self.assertTrue(a < b, '"%s" is not less than "%s"' % (str(a), str(b)))
    self._dep_test(iter, assertLess, deps)