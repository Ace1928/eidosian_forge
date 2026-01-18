from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.core.base import SortComponents
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
def _transform_constraintData(self, logical_constraint, visitor, transBlocks):
    parent_block = logical_constraint.parent_block()
    xfrm_block = transBlocks.get(parent_block)
    if xfrm_block is None:
        xfrm_block = self._create_transformation_block(parent_block)
        transBlocks[parent_block] = xfrm_block
    visitor.constraints = xfrm_block.transformed_constraints
    visitor.z_vars = xfrm_block.auxiliary_vars
    visitor.disjuncts = xfrm_block.auxiliary_disjuncts
    visitor.disjunctions = xfrm_block.auxiliary_disjunctions
    visitor.walk_expression(logical_constraint.expr)
    logical_constraint.deactivate()