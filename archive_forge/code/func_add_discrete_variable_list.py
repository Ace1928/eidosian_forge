from pyomo.core import (
from pyomo.core.base import TransformationFactory, Suffix, ConstraintList, Integers
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import (
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.util.vars_from_expressions import get_vars_from_components
def add_discrete_variable_list(util_block):
    lst = util_block.discrete_variable_list = []
    for v in util_block.algebraic_variable_list:
        if v.is_integer():
            lst.append(v)