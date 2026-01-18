import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.log import LoggingIntercept
class TestNumpyRegistration(unittest.TestCase):

    def test_deprecation(self):
        with LoggingIntercept() as LOG:
            import pyomo.core.kernel.register_numpy_types as rnt
        self.assertRegex(LOG.getvalue(), 'DEPRECATED: pyomo.core.kernel.register_numpy_types is deprecated.')
        self.assertEqual(sorted(rnt.numpy_bool_names), sorted(numpy_bool_names))
        self.assertEqual(sorted(rnt.numpy_int_names), sorted(numpy_int_names))
        self.assertEqual(sorted(rnt.numpy_float_names), sorted(numpy_float_names))
        self.assertEqual(sorted(rnt.numpy_complex_names), sorted(numpy_complex_names))