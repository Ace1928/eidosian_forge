import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
def _get_model_nlp_interface(halt_on_evaluation_error=None):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], initialize=1.0)
    m.obj = pyo.Objective(expr=m.x[1] * pyo.sqrt(m.x[2]) + m.x[1] * m.x[3])
    m.eq1 = pyo.Constraint(expr=m.x[1] * pyo.sqrt(m.x[2]) == 1.0)
    nlp = PyomoNLP(m)
    interface = CyIpoptNLP(nlp, halt_on_evaluation_error=halt_on_evaluation_error)
    bad_primals = np.array([1.0, -2.0, 3.0])
    indices = nlp.get_primal_indices([m.x[1], m.x[2], m.x[3]])
    bad_primals = bad_primals[indices]
    return (m, nlp, interface, bad_primals)