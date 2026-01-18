import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def _x_to_nlp(self, m, nlp, values):
    indices = nlp.get_primal_indices([m.x[0], m.x[1], m.x[2], m.x[3]])
    reordered_values = [None for _ in m.x]
    for i, val in zip(indices, values):
        reordered_values[i] = val
    return reordered_values