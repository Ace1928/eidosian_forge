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
def _intX(m, x, t):
    return (m.y[x, t] - m.y0[x]) ** 2 + m.omega * m.u[x, t] ** 2