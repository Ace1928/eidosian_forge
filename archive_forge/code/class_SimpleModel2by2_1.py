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
class SimpleModel2by2_1(object):
    """
    The purpose of this model is to test second derivative computation
    when the external model is nonlinear only in x. This exercises
    the first term in the second derivative implicit function theorem.
    """

    def make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1], initialize=2.0)
        m.y = pyo.Var([0, 1], initialize=2.0)

        def residual_eqn_rule(m, i):
            if i == 0:
                return m.x[0] ** 2 + m.x[0] * m.y[0] + m.y[0] ** 2 == 1.0
            elif i == 1:
                return m.x[1] ** 2 + m.x[1] * m.y[1] == 2.0
        m.residual_eqn = pyo.Constraint([0, 1], rule=residual_eqn_rule)

        def external_eqn_rule(m, i):
            if i == 0:
                return m.y[0] + m.y[1] + m.x[0] * m.x[1] + m.x[0] ** 2 == 1.0
            elif i == 1:
                return m.y[0] + 2.0 * m.x[0] * m.x[1] + m.x[1] ** 2 == 2.0
        m.external_eqn = pyo.Constraint([0, 1], rule=external_eqn_rule)
        return m

    def evaluate_residual(self, x):
        f0 = x[0] ** 2 + 2 * x[0] - 2 * x[0] ** 2 * x[1] - x[1] ** 2 * x[0] + 4 - 8 * x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[0] ** 2 * x[1] ** 2 + 4 * x[0] * x[1] ** 3 + x[1] ** 4 - 1.0
        f1 = x[1] ** 2 - x[1] + x[0] * x[1] ** 2 + x[1] ** 3 - x[0] ** 2 * x[1] - 2.0
        return (f0, f1)

    def evaluate_jacobian(self, x):
        df0dx0 = 2 * x[0] + 2 - 4 * x[0] * x[1] - x[1] ** 2 - 8 * x[1] + 8 * x[0] * x[1] ** 2 + 4 * x[1] ** 3
        df0dx1 = -2 * x[0] ** 2 - 2 * x[0] * x[1] - 8 * x[0] - 8 * x[1] + 8 * x[0] ** 2 * x[1] + 12 * x[0] * x[1] ** 2 + 4 * x[1] ** 3
        df1dx0 = x[1] ** 2 - 2 * x[0] * x[1]
        df1dx1 = 2 * x[1] - 1 + 2 * x[0] * x[1] - x[0] ** 2 + 3 * x[1] ** 2
        return np.array([[df0dx0, df0dx1], [df1dx0, df1dx1]])

    def evaluate_hessian(self, x):
        df0dx0dx0 = 2 - 4 * x[1] + 8 * x[1] ** 2
        df0dx0dx1 = -4 * x[0] - 2 * x[1] - 8 + 16 * x[0] * x[1] + 12 * x[1] ** 2
        df0dx1dx1 = -2 * x[0] - 8 + 8 * x[0] ** 2 + 24 * x[0] * x[1] + 12 * x[1] ** 2
        df1dx0dx0 = -2 * x[1]
        df1dx0dx1 = 2 * x[1] - 2 * x[0]
        df1dx1dx1 = 2 + 2 * x[0] + 6 * x[1]
        d2f0 = np.array([[df0dx0dx0, df0dx0dx1], [df0dx0dx1, df0dx1dx1]])
        d2f1 = np.array([[df1dx0dx0, df1dx0dx1], [df1dx0dx1, df1dx1dx1]])
        return [d2f0, d2f1]

    def evaluate_external_variables(self, x):
        y0 = 2.0 - 2.0 * x[0] * x[1] - x[1] ** 2
        y1 = 1.0 - y0 - x[0] * x[1] - x[0] ** 2
        return (y0, y1)

    def evaluate_external_jacobian(self, x):
        dy0dx0 = -2.0 * x[1]
        dy0dx1 = -2.0 * x[0] - 2.0 * x[1]
        dy1dx0 = -dy0dx0 - x[1] - 2.0 * x[0]
        dy1dx1 = -dy0dx1 - x[0]
        return np.array([[dy0dx0, dy0dx1], [dy1dx0, dy1dx1]])

    def evaluate_external_hessian(self, x):
        dy0dx0dx0 = 0.0
        dy0dx0dx1 = -2.0
        dy0dx1dx1 = -2.0
        dy1dx0dx0 = -dy0dx0dx0 - 2.0
        dy1dx0dx1 = -dy0dx0dx1 - 1.0
        dy1dx1dx1 = -dy0dx1dx1
        dy0dxdx = np.array([[dy0dx0dx0, dy0dx0dx1], [dy0dx0dx1, dy0dx1dx1]])
        dy1dxdx = np.array([[dy1dx0dx0, dy1dx0dx1], [dy1dx0dx1, dy1dx1dx1]])
        return [dy0dxdx, dy1dxdx]