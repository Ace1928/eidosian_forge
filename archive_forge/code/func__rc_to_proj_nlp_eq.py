import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def _rc_to_proj_nlp_eq(self, m, nlp, rc):
    var_indices = [1, 0]
    con_indices = nlp.get_equality_constraint_indices([m.eq_con_1, m.eq_con_2])
    i, j = rc
    return (con_indices[i], var_indices[j])