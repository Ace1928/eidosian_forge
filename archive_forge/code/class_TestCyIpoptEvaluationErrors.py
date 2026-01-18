import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
class TestCyIpoptEvaluationErrors(unittest.TestCase):

    @unittest.skipUnless(cyipopt_ge_1_3, 'cyipopt version < 1.3.0')
    def test_error_in_objective(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=False)
        msg = 'Error in objective function'
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.objective(bad_x)

    def test_error_in_objective_halt(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=True)
        msg = 'Error in AMPL evaluation'
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.objective(bad_x)

    @unittest.skipUnless(cyipopt_ge_1_3, 'cyipopt version < 1.3.0')
    def test_error_in_gradient(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=False)
        msg = 'Error in objective gradient'
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.gradient(bad_x)

    def test_error_in_gradient_halt(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=True)
        msg = 'Error in AMPL evaluation'
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.gradient(bad_x)

    @unittest.skipUnless(cyipopt_ge_1_3, 'cyipopt version < 1.3.0')
    def test_error_in_constraints(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=False)
        msg = 'Error in constraint evaluation'
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.constraints(bad_x)

    def test_error_in_constraints_halt(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=True)
        msg = 'Error in AMPL evaluation'
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.constraints(bad_x)

    @unittest.skipUnless(cyipopt_ge_1_3, 'cyipopt version < 1.3.0')
    def test_error_in_jacobian(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=False)
        msg = 'Error in constraint Jacobian'
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.jacobian(bad_x)

    def test_error_in_jacobian_halt(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=True)
        msg = 'Error in AMPL evaluation'
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.jacobian(bad_x)

    @unittest.skipUnless(cyipopt_ge_1_3, 'cyipopt version < 1.3.0')
    def test_error_in_hessian(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=False)
        msg = 'Error in Lagrangian Hessian'
        with self.assertRaisesRegex(cyipopt.CyIpoptEvaluationError, msg):
            interface.hessian(bad_x, [1.0], 0.0)

    def test_error_in_hessian_halt(self):
        m, nlp, interface, bad_x = _get_model_nlp_interface(halt_on_evaluation_error=True)
        msg = 'Error in AMPL evaluation'
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            interface.hessian(bad_x, [1.0], 0.0)