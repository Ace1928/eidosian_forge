import datetime
import multiprocessing
import os
import time
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Param
class TestPyomoUnittest(unittest.TestCase):

    def test_assertStructuredAlmostEqual_comparison(self):
        a = 1
        b = 1
        self.assertStructuredAlmostEqual(a, b)
        b -= 9.999e-08
        self.assertStructuredAlmostEqual(a, b)
        b -= 9.999e-08
        with self.assertRaisesRegex(self.failureException, '1 !~= 0.9999'):
            self.assertStructuredAlmostEqual(a, b)
        b = 1
        self.assertStructuredAlmostEqual(a, b, reltol=1e-06)
        b -= 9.999e-07
        self.assertStructuredAlmostEqual(a, b, reltol=1e-06)
        b -= 9.999e-07
        with self.assertRaisesRegex(self.failureException, '1 !~= 0.999'):
            self.assertStructuredAlmostEqual(a, b, reltol=1e-06)
        b = 1
        self.assertStructuredAlmostEqual(a, b, places=6)
        b -= 9.999e-07
        self.assertStructuredAlmostEqual(a, b, places=6)
        b -= 9.999e-07
        with self.assertRaisesRegex(self.failureException, '1 !~= 0.999'):
            self.assertStructuredAlmostEqual(a, b, places=6)
        with self.assertRaisesRegex(self.failureException, '10 !~= 10.01'):
            self.assertStructuredAlmostEqual(10, 10.01, abstol=0.001)
        self.assertStructuredAlmostEqual(10, 10.01, reltol=0.001)
        with self.assertRaisesRegex(self.failureException, '10 !~= 10.01'):
            self.assertStructuredAlmostEqual(10, 10.01, delta=0.001)

    def test_assertStructuredAlmostEqual_nan(self):
        a = float('nan')
        b = float('nan')
        self.assertStructuredAlmostEqual(a, b)

    def test_assertStructuredAlmostEqual_errorChecking(self):
        with self.assertRaisesRegex(ValueError, 'Cannot specify more than one of {places, delta, abstol}'):
            self.assertStructuredAlmostEqual(1, 1, places=7, delta=1)

    def test_assertStructuredAlmostEqual_str(self):
        self.assertStructuredAlmostEqual('hi', 'hi')
        with self.assertRaisesRegex(self.failureException, "'hi' !~= 'hello'"):
            self.assertStructuredAlmostEqual('hi', 'hello')
        with self.assertRaisesRegex(self.failureException, "'hi' !~= \\['h',"):
            self.assertStructuredAlmostEqual('hi', ['h', 'i'])

    def test_assertStructuredAlmostEqual_othertype(self):
        a = datetime.datetime(1, 1, 1)
        b = datetime.datetime(1, 1, 1)
        self.assertStructuredAlmostEqual(a, b)
        b = datetime.datetime(1, 1, 2)
        with self.assertRaisesRegex(self.failureException, 'datetime.* !~= datetime'):
            self.assertStructuredAlmostEqual(a, b)

    def test_assertStructuredAlmostEqual_list(self):
        a = [1, 2]
        b = [1, 2, 3]
        with self.assertRaisesRegex(self.failureException, 'sequences are different sizes \\(2 != 3\\)'):
            self.assertStructuredAlmostEqual(a, b)
        self.assertStructuredAlmostEqual(a, b, allow_second_superset=True)
        a.append(3)
        self.assertStructuredAlmostEqual(a, b)
        b[1] -= 1.999e-07
        self.assertStructuredAlmostEqual(a, b)
        b[1] -= 1.999e-07
        with self.assertRaisesRegex(self.failureException, '2 !~= 1.999'):
            self.assertStructuredAlmostEqual(a, b)

    def test_assertStructuredAlmostEqual_dict(self):
        a = {1: 2, 3: 4}
        b = {1: 2, 3: 4, 5: 6}
        with self.assertRaisesRegex(self.failureException, 'mappings are different sizes \\(2 != 3\\)'):
            self.assertStructuredAlmostEqual(a, b)
        self.assertStructuredAlmostEqual(a, b, allow_second_superset=True)
        a[5] = 6
        self.assertStructuredAlmostEqual(a, b)
        b[1] -= 1.999e-07
        self.assertStructuredAlmostEqual(a, b)
        b[1] -= 1.999e-07
        with self.assertRaisesRegex(self.failureException, '2 !~= 1.999'):
            self.assertStructuredAlmostEqual(a, b)
        del b[1]
        b[6] = 6
        with self.assertRaisesRegex(self.failureException, 'key \\(1\\) from first not found in second'):
            self.assertStructuredAlmostEqual(a, b)

    def test_assertStructuredAlmostEqual_nested(self):
        a = {1.1: [1, 2, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
        b = {1.1: [1, 2, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
        self.assertStructuredAlmostEqual(a, b)
        b[1.1][2] -= 1.999e-07
        b[3][1] -= 9.999e-08
        self.assertStructuredAlmostEqual(a, b)
        b[1.1][2] -= 1.999e-07
        with self.assertRaisesRegex(self.failureException, '3 !~= 2.999'):
            self.assertStructuredAlmostEqual(a, b)

    def test_assertStructuredAlmostEqual_numericvalue(self):
        m = ConcreteModel()
        m.v = Var(initialize=2.0)
        m.p = Param(initialize=2.0)
        a = {1.1: [1, m.p, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
        b = {1.1: [1, m.v, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
        self.assertStructuredAlmostEqual(a, b)
        m.v.set_value(m.v.value - 1.999e-07)
        self.assertStructuredAlmostEqual(a, b)
        m.v.set_value(m.v.value - 1.999e-07)
        with self.assertRaisesRegex(self.failureException, '2.0 !~= 1.999'):
            self.assertStructuredAlmostEqual(a, b)

    def test_timeout_fcn_call(self):
        self.assertEqual(short_sleep(), 42)
        with self.assertRaisesRegex(TimeoutError, 'test timed out after 0.01 seconds'):
            long_sleep()
        with self.assertRaisesRegex(NameError, "name 'foo' is not defined\\s+Original traceback:"):
            raise_exception()
        with self.assertRaisesRegex(AssertionError, '^0 != 1$'):
            fail()

    @unittest.timeout(10)
    def test_timeout(self):
        self.assertEqual(0, 0)

    @unittest.expectedFailure
    @unittest.timeout(0.01)
    def test_timeout_timeout(self):
        time.sleep(1)
        self.assertEqual(0, 1)

    @unittest.timeout(10)
    def test_timeout_skip(self):
        if TestPyomoUnittest.test_timeout_skip.skip:
            self.skipTest('Skipping this test')
        self.assertEqual(0, 1)
    test_timeout_skip.skip = True

    def test_timeout_skip_fails(self):
        try:
            with self.assertRaisesRegex(unittest.SkipTest, 'Skipping this test'):
                self.test_timeout_skip()
            TestPyomoUnittest.test_timeout_skip.skip = False
            with self.assertRaisesRegex(AssertionError, '0 != 1'):
                self.test_timeout_skip()
        finally:
            TestPyomoUnittest.test_timeout_skip.skip = True

    @unittest.timeout(10)
    def bound_function(self):
        self.assertEqual(0, 0)

    def test_bound_function(self):
        if multiprocessing.get_start_method() == 'fork':
            self.bound_function()
            return
        LOG = StringIO()
        with LoggingIntercept(LOG):
            with self.assertRaises((TypeError, EOFError, AttributeError)):
                self.bound_function()
        self.assertIn("platform that does not support 'fork'", LOG.getvalue())
        self.assertIn('one of its arguments is not serializable', LOG.getvalue())

    @unittest.timeout(10, require_fork=True)
    def bound_function_require_fork(self):
        self.assertEqual(0, 0)

    def test_bound_function_require_fork(self):
        if multiprocessing.get_start_method() == 'fork':
            self.bound_function_require_fork()
            return
        with self.assertRaisesRegex(unittest.SkipTest, 'timeout requires unavailable fork interface'):
            self.bound_function_require_fork()