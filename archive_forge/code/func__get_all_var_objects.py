import itertools
import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base import Reference, TransformationFactory
import pyomo.core.expr as EXPR
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import get_gdp_tree, _to_dict
from pyomo.network import Port
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.repn import generate_standard_repn
from weakref import ref as weakref_ref
def _get_all_var_objects(self, active_disjuncts):
    seen = set()
    for disj in active_disjuncts:
        for constraint in disj.component_data_objects(Constraint, active=True, sort=SortComponents.deterministic, descend_into=Block):
            for var in EXPR.identify_variables(constraint.expr, include_fixed=True):
                if id(var) not in seen:
                    seen.add(id(var))
                    yield var