import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
@unittest.skipIf(cyipopt_available, 'cyipopt is available')
class TestCyIpoptNotAvailable(unittest.TestCase):

    def test_not_available_exception(self):
        model = create_model1()
        nlp = PyomoNLP(model)
        msg = 'cyipopt is required'
        with self.assertRaisesRegex(RuntimeError, msg):
            solver = CyIpoptSolver(CyIpoptNLP(nlp))