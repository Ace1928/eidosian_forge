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
def _update_bounds_dict(self, v_bounds, lower, upper, disjunct, gdp_forest):
    lb, ub = self._get_tightest_ancestral_bounds(v_bounds, disjunct, gdp_forest)
    if lower is not None:
        if lb is None or lower > lb:
            lb = lower
    if upper is not None:
        if ub is None or upper < ub:
            ub = upper
    v_bounds[disjunct] = (lb, ub)