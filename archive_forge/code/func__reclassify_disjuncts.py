from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def _reclassify_disjuncts(self, true_disjuncts, false_disjuncts, reverse_dict, disjunct_set, disjunct_containers):
    for disj in true_disjuncts:
        reverse_dict[disj] = (disj.indicator_var.fixed, disj.indicator_var.value)
        self._update_transformed_disjuncts(disj, disjunct_set, disjunct_containers)
        parent_block = disj.parent_block()
        parent_block.reclassify_component_type(disj, Block)
        disj.indicator_var.fix(True)
    for disj in false_disjuncts:
        reverse_dict[disj] = (disj.indicator_var.fixed, disj.indicator_var.value)
        self._update_transformed_disjuncts(disj, disjunct_set, disjunct_containers)
        disj.deactivate()