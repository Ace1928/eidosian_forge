import logging
from collections import defaultdict
from pyomo.common.autoslots import AutoSlots
import pyomo.common.config as cfg
from pyomo.common import deprecated
from pyomo.common.collections import ComponentMap, ComponentSet, DefaultComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numvalue import ZeroConstant
import pyomo.core.expr as EXPR
from pyomo.core.base import TransformationFactory, Reference
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.disjunct import _DisjunctData
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.util.vars_from_expressions import get_vars_from_components
from weakref import ref as weakref_ref
def _add_transformation_block(self, to_block):
    transBlock, new_block = super()._add_transformation_block(to_block)
    if not new_block:
        return (transBlock, new_block)
    transBlock.lbub = Set(initialize=['lb', 'ub', 'eq'])
    transBlock.disaggregationConstraints = Constraint(NonNegativeIntegers)
    transBlock._disaggregationConstraintMap = ComponentMap()
    transBlock._disaggregatedVars = Var(NonNegativeIntegers, dense=False)
    transBlock._boundsConstraints = Constraint(NonNegativeIntegers, transBlock.lbub)
    return (transBlock, True)