import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
def _make_external_model():
    m = pyo.ConcreteModel()
    m.a = pyo.Var()
    m.b = pyo.Var()
    m.r = pyo.Var()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.x_out = pyo.Var()
    m.y_out = pyo.Var()
    m.c_out_1 = pyo.Constraint(expr=m.x_out - m.x == 0)
    m.c_out_2 = pyo.Constraint(expr=m.y_out - m.y == 0)
    m.c_ex_1 = pyo.Constraint(expr=m.x ** 3 - 2 * m.y == m.a ** 2 + m.b ** 3 - m.r ** 3 - 2)
    m.c_ex_2 = pyo.Constraint(expr=m.x + m.y ** 3 == m.a ** 3 + 2 * m.b ** 2 + m.r ** 2 + 1)
    return m