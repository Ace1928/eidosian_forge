import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
@unittest.skipIf(not parmest.parmest_available, 'Cannot test parmest: required dependencies are missing')
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestSemibatchExamples(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_model(self):
        from pyomo.contrib.parmest.examples.semibatch import semibatch
        semibatch.main()

    def test_parameter_estimation_example(self):
        from pyomo.contrib.parmest.examples.semibatch import parameter_estimation_example
        parameter_estimation_example.main()

    def test_scenario_example(self):
        from pyomo.contrib.parmest.examples.semibatch import scenario_example
        scenario_example.main()