import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def _rc_to_nlp(self, m, nlp, rc):
    var_indices = nlp.get_primal_indices(list(m.x.values()))
    con_indices = nlp.get_constraint_indices([m.eq_con_1, m.eq_con_2, m.ineq_con_1, m.ineq_con_2, m.ineq_con_3])
    i, j = rc
    return (con_indices[i], var_indices[j])