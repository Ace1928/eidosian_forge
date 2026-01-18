from pyomo.core import (
from pyomo.core.base import TransformationFactory, Suffix, ConstraintList, Integers
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import (
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.util.vars_from_expressions import get_vars_from_components
def add_disjunct_list(util_block):
    model = util_block.parent_block()
    util_block.disjunct_list = list(model.component_data_objects(ctype=Disjunct, active=True, descend_into=(Block, Disjunct), sort=SortComponents.deterministic))