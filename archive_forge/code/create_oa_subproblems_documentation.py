from pyomo.core import (
from pyomo.core.base import TransformationFactory, Suffix, ConstraintList, Integers
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import (
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.util.vars_from_expressions import get_vars_from_components
Clone the original, and reclassify all the Disjuncts to Blocks.
    We'll also call logical_to_disjunctive and bigm the disjunctive parts in
    case any of the indicator_vars are used in logical constraints and to make
    sure that the rest of the model is algebraic (assuming it was a proper
    GDP to begin with).
    