import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
class ImplicitFunctionNoInputs(ImplicitFunctionInputsDontAppear):
    """The same system as with inputs that don't appear, but now the
    inputs are not provided to the implicit function solver

    """

    def get_parameters(self):
        return []

    def get_input_output_sequence(self):
        inputs = [()]
        outputs = [(2.498253, -0.569676, 0.821869)]
        return list(zip(inputs, outputs))