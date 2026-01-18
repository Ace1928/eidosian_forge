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
def _process_container(blk, config):
    if not hasattr(blk, '_induced_linearity_info'):
        blk._induced_linearity_info = Block()
    else:
        assert blk._induced_linearity_info.ctype == Block
    eff_discr_vars = detect_effectively_discrete_vars(blk, config.equality_tolerance)
    possible_var_values = determine_valid_values(blk, eff_discr_vars, config)
    bilinear_map = _bilinear_expressions(blk)
    processed_pairs = ComponentSet()
    for v1, var_values in possible_var_values.items():
        v1_pairs = bilinear_map.get(v1, ())
        for v2, bilinear_constrs in v1_pairs.items():
            if (v1, v2) in processed_pairs:
                continue
            _process_bilinear_constraints(blk, v1, v2, var_values, bilinear_constrs)
            processed_pairs.add((v2, v1))