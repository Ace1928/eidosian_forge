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
class SimpleModel1(object):

    def make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=2.0)
        m.residual_eqn = pyo.Constraint(expr=m.x ** 2 + m.y ** 2 == 1.0)
        m.external_eqn = pyo.Constraint(expr=m.x * m.y == 0.2)
        return m

    def evaluate_external_variables(self, x):
        return 0.2 / x

    def evaluate_external_jacobian(self, x):
        return -0.2 / x ** 2

    def evaluate_external_hessian(self, x):
        return 0.4 / x ** 3

    def evaluate_residual(self, x):
        return x ** 2 + 0.04 / x ** 2 - 1

    def evaluate_jacobian(self, x):
        return 2 * x - 0.08 / x ** 3

    def evaluate_hessian(self, x):
        return 2 + 0.24 / x ** 4