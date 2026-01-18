from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.expr import identify_variables
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import is_child_of, get_gdp_tree
from pyomo.repn.standard_repn import generate_standard_repn
import logging
def _update_bounds_from_constraints(self, disjunct, bound_dict, gdp_forest, is_root=False):
    for constraint in disjunct.component_data_objects(Constraint, active=True, descend_into=Block, sort=SortComponents.deterministic):
        var_gen = identify_variables(constraint.body, include_fixed=False)
        try:
            next(var_gen)
        except StopIteration:
            continue
        try:
            next(var_gen)
        except StopIteration:
            repn = generate_standard_repn(constraint.body)
            if not repn.is_linear() or len(repn.linear_vars) != 1:
                continue
            v = repn.linear_vars[0]
            coef = repn.linear_coefs[0]
            constant = repn.constant
            lower = (value(constraint.lower) - constant) / coef if constraint.lower is not None else None
            upper = (value(constraint.upper) - constant) / coef if constraint.upper is not None else None
            if coef < 0:
                lower, upper = (upper, lower)
            v_bounds = self._get_bound_dict_for_var(bound_dict, v)
            self._update_bounds_dict(v_bounds, lower, upper, disjunct if not is_root else None, gdp_forest)
            if not is_root:
                if disjunct in v_bounds['to_deactivate']:
                    v_bounds['to_deactivate'][disjunct].add(constraint)
                else:
                    v_bounds['to_deactivate'][disjunct] = ComponentSet([constraint])