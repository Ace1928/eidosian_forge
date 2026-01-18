import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.core import (
from pyomo.core.base import TransformationFactory, Reference
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import is_child_of, _get_constraint_transBlock, _to_dict
from pyomo.core.util import target_list
from pyomo.network import Port
from pyomo.repn import generate_standard_repn
from weakref import ref as weakref_ref, ReferenceType
def _update_M_from_suffixes(self, constraint, suffix_list, lower, upper):
    need_lower = constraint.lower is not None and lower[0] is None
    need_upper = constraint.upper is not None and upper[0] is None
    M = None
    for bigm in suffix_list:
        if constraint in bigm:
            M = bigm[constraint]
            lower, upper, need_lower, need_upper = self._process_M_value(M, lower, upper, need_lower, need_upper, bigm, constraint, constraint)
            if not need_lower and (not need_upper):
                return (lower, upper)
        if constraint.parent_component() in bigm:
            parent = constraint.parent_component()
            M = bigm[parent]
            lower, upper, need_lower, need_upper = self._process_M_value(M, lower, upper, need_lower, need_upper, bigm, parent, constraint)
            if not need_lower and (not need_upper):
                return (lower, upper)
    if M is None:
        for bigm in suffix_list:
            if None in bigm:
                M = bigm[None]
                lower, upper, need_lower, need_upper = self._process_M_value(M, lower, upper, need_lower, need_upper, bigm, None, constraint)
            if not need_lower and (not need_upper):
                return (lower, upper)
    return (lower, upper)