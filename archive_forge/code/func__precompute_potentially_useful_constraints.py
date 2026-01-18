from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def _precompute_potentially_useful_constraints(transBlock_rHull, disaggregated_vars):
    instance_rHull = transBlock_rHull.model()
    constraints = transBlock_rHull.constraints_for_FME = []
    for constraint in instance_rHull.component_data_objects(Constraint, active=True, descend_into=Block, sort=SortComponents.deterministic):
        repn = generate_standard_repn(constraint.body)
        for v in repn.linear_vars + repn.quadratic_vars + repn.nonlinear_vars:
            if v in disaggregated_vars:
                constraints.append(constraint)
                break