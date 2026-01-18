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
Returns the largest M value for any constraint on the model.

        Parameters
        ----------
        model: A GDP model that has been transformed with BigM
        