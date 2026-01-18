import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.visitor import (
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import SolverFactory
import itertools as it
import timeit
from contextlib import contextmanager
import logging
import math
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.log import Preformatted
def add_decision_rule_variables(model_data, config):
    """
    Add variables for polynomial decision rules to the working
    model.

    Parameters
    ----------
    model_data : ROSolveResults
        Model data.
    config : config_dict
        PyROS solver options.

    Note
    ----
    Decision rule variables are considered first-stage decision
    variables which do not get copied at each iteration.
    PyROS currently supports static (zeroth order),
    affine (first-order), and quadratic DR.
    """
    second_stage_variables = model_data.working_model.util.second_stage_variables
    first_stage_variables = model_data.working_model.util.first_stage_variables
    decision_rule_vars = []
    degree = config.decision_rule_order
    num_uncertain_params = len(model_data.working_model.util.uncertain_params)
    num_dr_vars = sp.special.comb(N=num_uncertain_params + degree, k=degree, exact=True, repetition=False)
    for idx, ss_var in enumerate(second_stage_variables):
        indexed_dr_var = Var(range(num_dr_vars), initialize=0, bounds=(None, None), domain=Reals)
        model_data.working_model.add_component(f'decision_rule_var_{idx}', indexed_dr_var)
        indexed_dr_var[0].set_value(value(ss_var, exception=False))
        first_stage_variables.extend(indexed_dr_var.values())
        decision_rule_vars.append(indexed_dr_var)
    model_data.working_model.util.decision_rule_vars = decision_rule_vars