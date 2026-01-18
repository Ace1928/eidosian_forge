import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
@unittest.skipIf(not parmest.parmest_available, 'Cannot test parmest: required dependencies are missing')
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestReactionKineticsExamples(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_example(self):
        from pyomo.contrib.parmest.examples.reaction_kinetics import simple_reaction_parmest_example
        simple_reaction_parmest_example.main()