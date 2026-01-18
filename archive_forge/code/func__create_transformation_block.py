from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.core.base import SortComponents
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
def _create_transformation_block(self, context):
    new_xfrm_block_name = unique_component_name(context, '_logical_to_disjunctive')
    new_xfrm_block = Block(doc='Transformation objects for logical_to_disjunctive')
    context.add_component(new_xfrm_block_name, new_xfrm_block)
    new_xfrm_block.transformed_constraints = ConstraintList()
    new_xfrm_block.auxiliary_vars = VarList(domain=Binary)
    new_xfrm_block.auxiliary_disjuncts = Disjunct(NonNegativeIntegers)
    new_xfrm_block.auxiliary_disjunctions = Disjunction(NonNegativeIntegers)
    return new_xfrm_block