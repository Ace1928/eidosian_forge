import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
class ImplicitFunctionInputsDontAppear(object):
    """This is an implicit function designed to test the edge case
    where inputs do not appear in the system defining the implicit
    function (i.e. the function is constant).

    """

    def __init__(self):
        self._model = self._make_model()

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=[1, 2, 3])
        m.J = pyo.Set(initialize=[1, 2])
        m.x = pyo.Var(m.I, initialize=1.0)
        m.p = pyo.Var(m.J, initialize=1.0)
        m.con1 = pyo.Constraint(expr=m.x[2] ** 2 + m.x[3] ** 2 == 1.0)
        m.con2 = pyo.Constraint(expr=2 * m.x[1] + 3 * m.x[2] - 4 * m.x[3] == 0.0)
        m.con3 = pyo.Constraint(expr=1.0 == 2 * pyo.exp(m.x[2] / m.x[3]))
        m.obj = pyo.Objective(expr=0.0)
        return m

    def get_parameters(self):
        m = self._model
        return [m.p[1], m.p[2]]

    def get_variables(self):
        m = self._model
        return [m.x[1], m.x[2], m.x[3]]

    def get_equations(self):
        m = self._model
        return [m.con1, m.con2, m.con3]

    def get_input_output_sequence(self):
        p1_inputs = [-1.0, 0.0]
        p2_inputs = [1.0]
        inputs = list(itertools.product(p1_inputs, p2_inputs))
        outputs = [(2.498253, -0.569676, 0.821869), (2.498253, -0.569676, 0.821869)]
        return list(zip(inputs, outputs))