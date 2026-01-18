import copy
import itertools
from pyomo.core import Block, ConstraintList, Set, Constraint
from pyomo.core.base import Reference
from pyomo.common.modeling import unique_component_name
from pyomo.gdp.disjunct import Disjunct, Disjunction
import logging
Squish all singleton tuples into their non-tuple values.