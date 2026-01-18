from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def _reverse_transform_disjunction(self, disjunction, reverse_token, disjunct_set, disjunct_containers):
    if disjunction in reverse_token['_disjunctions']:
        disjunction.activate()
    for disjunct in disjunction.disjuncts:
        if disjunct in reverse_token['_disjuncts']:
            fixed, val = reverse_token['_disjuncts'][disjunct]
            disjunct.parent_block().reclassify_component_type(disjunct, Disjunct)
            disjunct.activate()
            disjunct.indicator_var = val
            disjunct.indicator_var.fixed = fixed