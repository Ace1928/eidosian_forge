from pyomo.contrib.pynumero.sparse import BlockMatrix
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import pyomo.environ as pyo
import pyomo.dae as dae
def _int_rule(M, i):
    return M.x[1, i] ** 2 + M.x[2, i] ** 2 + M.u[i] ** 2