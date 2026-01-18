import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.pyomo_ext_cyipopt import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
class PressureDropModel(ExternalInputOutputModel):

    def __init__(self):
        self._Pin = None
        self._c1 = None
        self._c2 = None
        self._F = None

    def set_inputs(self, input_values):
        assert len(input_values) == 4
        self._Pin = input_values[0]
        self._c1 = input_values[1]
        self._c2 = input_values[2]
        self._F = input_values[3]

    def evaluate_outputs(self):
        P1 = self._Pin - self._c1 * self._F ** 2
        P2 = P1 - self._c2 * self._F ** 2
        return np.asarray([P1, P2], dtype=np.float64)

    def evaluate_derivatives(self):
        jac = [[1, -self._F ** 2, 0, -2 * self._c1 * self._F], [1, -self._F ** 2, -self._F ** 2, -2 * self._F * (self._c1 + self._c2)]]
        jac = np.asarray(jac, dtype=np.float64)
        return spa.coo_matrix(jac)