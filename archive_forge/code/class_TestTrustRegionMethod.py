import logging
from io import StringIO
import sys
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.trustregion.examples import example1, example2
from pyomo.environ import SolverFactory
@unittest.skipIf(not SolverFactory('ipopt').available(False), 'The IPOPT solver is not available')
class TestTrustRegionMethod(unittest.TestCase):

    def test_example1(self):
        log_OUTPUT = StringIO()
        print_OUTPUT = StringIO()
        sys.stdout = print_OUTPUT
        with LoggingIntercept(log_OUTPUT, 'pyomo.contrib.trustregion', logging.INFO):
            example1.main()
        sys.stdout = sys.__stdout__
        self.assertIn('Iteration 0', log_OUTPUT.getvalue())
        self.assertIn('Iteration 4', log_OUTPUT.getvalue())
        self.assertNotIn('Iteration 5', log_OUTPUT.getvalue())
        self.assertIn('theta-type step', log_OUTPUT.getvalue())
        self.assertNotIn('f-type step', log_OUTPUT.getvalue())
        self.assertNotIn('EXIT: Optimal solution found.', log_OUTPUT.getvalue())
        self.assertNotIn('None :   True : 0.2770447887637415', log_OUTPUT.getvalue())
        self.assertIn('Iteration 0', print_OUTPUT.getvalue())
        self.assertIn('Iteration 4', print_OUTPUT.getvalue())
        self.assertNotIn('Iteration 5', print_OUTPUT.getvalue())
        self.assertIn('theta-type step', print_OUTPUT.getvalue())
        self.assertNotIn('f-type step', print_OUTPUT.getvalue())
        self.assertIn('EXIT: Optimal solution found.', print_OUTPUT.getvalue())
        self.assertIn('None :   True : 0.2770447887637415', print_OUTPUT.getvalue())

    def test_example2(self):
        log_OUTPUT = StringIO()
        print_OUTPUT = StringIO()
        sys.stdout = print_OUTPUT
        with LoggingIntercept(log_OUTPUT, 'pyomo.contrib.trustregion', logging.INFO):
            example2.main()
        sys.stdout = sys.__stdout__
        self.assertIn('Iteration 0', log_OUTPUT.getvalue())
        self.assertIn('Iteration 70', log_OUTPUT.getvalue())
        self.assertNotIn('Iteration 85', log_OUTPUT.getvalue())
        self.assertIn('theta-type step', log_OUTPUT.getvalue())
        self.assertIn('f-type step', log_OUTPUT.getvalue())
        self.assertIn('step rejected', log_OUTPUT.getvalue())
        self.assertNotIn('EXIT: Optimal solution found.', log_OUTPUT.getvalue())
        self.assertNotIn('None :   True : 48.383116936949', log_OUTPUT.getvalue())
        self.assertIn('Iteration 0', print_OUTPUT.getvalue())
        self.assertIn('Iteration 70', print_OUTPUT.getvalue())
        self.assertNotIn('Iteration 85', print_OUTPUT.getvalue())
        self.assertIn('theta-type step', print_OUTPUT.getvalue())
        self.assertIn('f-type step', print_OUTPUT.getvalue())
        self.assertIn('step rejected', print_OUTPUT.getvalue())
        self.assertIn('EXIT: Optimal solution found.', print_OUTPUT.getvalue())
        self.assertIn('None :   True : 48.383116936949', print_OUTPUT.getvalue())