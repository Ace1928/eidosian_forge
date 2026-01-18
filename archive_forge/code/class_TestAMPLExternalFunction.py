import os
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.common.gsl import find_GSL
from pyomo.environ import ConcreteModel, ExternalFunction, Var, Objective
class TestAMPLExternalFunction(unittest.TestCase):

    def assertListsAlmostEqual(self, first, second, places=7, msg=None):
        self.assertEqual(len(first), len(second))
        msg = 'lists %s and %s differ at item ' % (first, second)
        for i, a in enumerate(first):
            self.assertAlmostEqual(a, second[i], places, msg + str(i))

    def test_solve_gsl_function(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find the amplgsl.dll library')
        model = ConcreteModel()
        model.z_func = ExternalFunction(library=DLL, function='gsl_sf_gamma')
        model.x = Var(initialize=3, bounds=(1e-05, None))
        model.o = Objective(expr=model.z_func(model.x))
        nlp = PyomoNLP(model)
        self.assertAlmostEqual(nlp.evaluate_objective(), 2, 7)
        assert 'AMPLFUNC' not in os.environ