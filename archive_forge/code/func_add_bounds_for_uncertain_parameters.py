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
def add_bounds_for_uncertain_parameters(model, config):
    """
    This function solves a set of optimization problems to determine bounds on the uncertain parameters
    given the uncertainty set description. These bounds will be added as additional constraints to the uncertainty_set_constr
    constraint. Should only be called once set_as_constraint() has been called on the separation_model object.
    :param separation_model: the model on which to add the bounds
    :param config: solver config
    :return:
    """
    uncertain_param_bounds = []
    bounding_model = ConcreteModel()
    bounding_model.util = Block()
    bounding_model.util.uncertain_param_vars = IndexedVar(model.util.uncertain_param_vars.index_set())
    for tup in model.util.uncertain_param_vars.items():
        bounding_model.util.uncertain_param_vars[tup[0]].set_value(tup[1].value, skip_validation=True)
    bounding_model.add_component('uncertainty_set_constraint', config.uncertainty_set.set_as_constraint(uncertain_params=bounding_model.util.uncertain_param_vars, model=bounding_model, config=config))
    for idx, param in enumerate(list(bounding_model.util.uncertain_param_vars.values())):
        bounding_model.add_component('lb_obj_' + str(idx), Objective(expr=param, sense=minimize))
        bounding_model.add_component('ub_obj_' + str(idx), Objective(expr=param, sense=maximize))
    for o in bounding_model.component_data_objects(Objective):
        o.deactivate()
    for i in range(len(bounding_model.util.uncertain_param_vars)):
        bounds = []
        for limit in ('lb', 'ub'):
            getattr(bounding_model, limit + '_obj_' + str(i)).activate()
            res = config.global_solver.solve(bounding_model, tee=False)
            bounds.append(bounding_model.util.uncertain_param_vars[i].value)
            getattr(bounding_model, limit + '_obj_' + str(i)).deactivate()
        uncertain_param_bounds.append(bounds)
    for idx, bound in enumerate(uncertain_param_bounds):
        model.util.uncertain_param_vars[idx].setlb(bound[0])
        model.util.uncertain_param_vars[idx].setub(bound[1])
    return