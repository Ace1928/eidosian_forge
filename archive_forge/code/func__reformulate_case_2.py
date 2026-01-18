import logging
import textwrap
from math import fabs
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.preprocessing.util import SuppressConstantObjectiveWarning
from pyomo.core import (
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
def _reformulate_case_2(blk, v1, v2, bilinear_constr):
    repn = generate_standard_repn(bilinear_constr.body)
    replace_index = next((i for i, var_tup in enumerate(repn.quadratic_vars) if var_tup[0] is v1 and var_tup[1] is v2 or (var_tup[0] is v2 and var_tup[1] is v1)))
    bilinear_constr.set_value((bilinear_constr.lower, sum((coef * repn.linear_vars[i] for i, coef in enumerate(repn.linear_coefs))) + repn.quadratic_coefs[replace_index] * sum((val * blk.v_increment[val] for val in blk.valid_values)) + sum((repn.quadratic_coefs[i] * var_tup[0] * var_tup[1] for i, var_tup in enumerate(repn.quadratic_vars) if not i == replace_index)) + repn.constant + zero_if_None(repn.nonlinear_expr), bilinear_constr.upper))