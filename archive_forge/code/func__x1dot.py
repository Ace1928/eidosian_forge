from pyomo.contrib.pynumero.sparse import BlockMatrix
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import pyomo.environ as pyo
import pyomo.dae as dae
def _x1dot(M, i):
    if i == M.t.first():
        return pyo.Constraint.Skip
    return M.xdot[1, i] == (1 - M.x[2, i] ** 2) * M.x[1, i] - M.x[2, i] + M.u[i]