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
def _get_bound_dict_for_var(self, bound_dict, v):
    v_bounds = bound_dict.get(v)
    if v_bounds is None:
        v_bounds = bound_dict[v] = {None: (v.lb, v.ub), 'to_deactivate': ComponentMap()}
    return v_bounds