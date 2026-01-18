from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import numpy as np
from scipy.sparse import tril
import pyomo.environ as pe
from pyomo import dae
from pyomo.common.timing import TicTocTimer
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverInterface, LinearSolverStatus
def _pde(m, x, t):
    if t == start_t or x == end_x or x == start_x:
        e = pe.Constraint.Skip
    else:
        e = m.dydt[x, t] - m.v * m.dydx2[x, t] + m.dydx[x, t] * m.y[x, t] == m.r + m.u[x, t]
    return e