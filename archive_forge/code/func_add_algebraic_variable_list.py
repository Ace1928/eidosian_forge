from pyomo.core import (
from pyomo.core.base import TransformationFactory, Suffix, ConstraintList, Integers
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import (
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.util.vars_from_expressions import get_vars_from_components
def add_algebraic_variable_list(util_block, name=None):
    """
    This collects variables from active Constraints and Objectives. It descends
    into Disjuncts, but does not collect any indicator variables that do not
    appear in algebraic constraints pre-transformation.
    """
    model = util_block.parent_block()
    if name is None:
        name = 'algebraic_variable_list'
    setattr(util_block, name, list(get_vars_from_components(model, ctype=(Constraint, Objective), descend_into=(Block, Disjunct), active=True, sort=SortComponents.deterministic)))