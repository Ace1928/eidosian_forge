import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
class TestScaling(unittest.TestCase):

    def con_3_body(self, x, y, u, v):
        return 100000.0 * x ** 2 + 10000.0 * y ** 2 + 10.0 * u ** 2 + 1.0 * v ** 2

    def con_3_rhs(self):
        return 20000.0

    def con_4_body(self, x, y, u, v):
        return 0.01 * x + 0.001 * y + 0.0001 * u + 0.0001 * v

    def con_4_rhs(self):
        return 0.0003

    def make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0)
        m.y = pyo.Var(initialize=1.0)
        m.u = pyo.Var(initialize=1.0)
        m.v = pyo.Var(initialize=1.0)
        m.con_1 = pyo.Constraint(expr=m.x * m.y == m.u)
        m.con_2 = pyo.Constraint(expr=m.x ** 2 * m.y ** 3 == m.v)
        m.con_3 = pyo.Constraint(expr=self.con_3_body(m.x, m.y, m.u, m.v) == self.con_3_rhs())
        m.con_4 = pyo.Constraint(expr=self.con_4_body(m.x, m.y, m.u, m.v) == self.con_4_rhs())
        epm_model = pyo.ConcreteModel()
        epm_model.x = pyo.Reference(m.x)
        epm_model.y = pyo.Reference(m.y)
        epm_model.u = pyo.Reference(m.u)
        epm_model.v = pyo.Reference(m.v)
        epm_model.epm = ExternalPyomoModel([m.u, m.v], [m.x, m.y], [m.con_3, m.con_4], [m.con_1, m.con_2])
        epm_model.obj = pyo.Objective(expr=m.x ** 2 + m.y ** 2 + m.u ** 2 + m.v ** 2)
        epm_model.egb = ExternalGreyBoxBlock()
        epm_model.egb.set_external_model(epm_model.epm, inputs=[m.u, m.v])
        return epm_model

    def test_get_set_scaling_factors(self):
        m = self.make_model()
        scaling_factors = [0.0001, 10000.0]
        m.epm.set_equality_constraint_scaling_factors(scaling_factors)
        epm_sf = m.epm.get_equality_constraint_scaling_factors()
        np.testing.assert_array_equal(scaling_factors, epm_sf)

    def test_pyomo_nlp(self):
        m = self.make_model()
        scaling_factors = [0.0001, 10000.0]
        m.epm.set_equality_constraint_scaling_factors(scaling_factors)
        nlp = PyomoNLPWithGreyBoxBlocks(m)
        nlp_sf = nlp.get_constraints_scaling()
        np.testing.assert_array_equal(scaling_factors, nlp_sf)

    @unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
    def test_cyipopt_nlp(self):
        m = self.make_model()
        scaling_factors = [0.0001, 10000.0]
        m.epm.set_equality_constraint_scaling_factors(scaling_factors)
        nlp = PyomoNLPWithGreyBoxBlocks(m)
        cyipopt_nlp = CyIpoptNLP(nlp)
        obj_scaling, x_scaling, g_scaling = cyipopt_nlp.scaling_factors()
        np.testing.assert_array_equal(scaling_factors, g_scaling)

    @unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
    def test_cyipopt_callback(self):
        m = self.make_model()
        scaling_factors = [0.0001, 10000.0]
        m.epm.set_equality_constraint_scaling_factors(scaling_factors)
        nlp = PyomoNLPWithGreyBoxBlocks(m)

        def callback(local_nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
            primals = tuple(local_nlp.get_primals())
            u, v, x, y = primals
            con_3_resid = scaling_factors[0] * abs(self.con_3_body(x, y, u, v) - self.con_3_rhs())
            con_4_resid = scaling_factors[1] * abs(self.con_4_body(x, y, u, v) - self.con_4_rhs())
            pred_inf_pr = max(con_3_resid, con_4_resid)
            self.assertAlmostEqual(inf_pr, pred_inf_pr)
        cyipopt_nlp = CyIpoptNLP(nlp, intermediate_callback=callback)
        x0 = nlp.get_primals()
        cyipopt = CyIpoptSolver(cyipopt_nlp, options={'max_iter': 0, 'nlp_scaling_method': 'user-scaling'})
        cyipopt.solve(x0=x0)