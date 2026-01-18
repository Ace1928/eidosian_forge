import pyomo.common.unittest as unittest
from pyomo.common.dependencies import scipy, scipy_available
import pyomo.environ as pyo
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
@unittest.skipUnless(AmplInterface.available(), 'AmplInterface is not available')
class TestSecantNewton(unittest.TestCase):

    def test_inherited_options_skipped(self):
        m, nlp = make_scalar_model()
        options = SecantNewtonNlpSolver.OPTIONS
        self.assertNotIn('maxiter', options)
        self.assertNotIn('secant', options)
        self.assertIn('secant_iter', options)
        self.assertIn('newton_iter', options)
        with self.assertRaisesRegex(ValueError, 'implicit.*keys are not allowed'):
            solver = SecantNewtonNlpSolver(nlp, options={'maxiter': 10})