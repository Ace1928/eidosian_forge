import copy
import itertools
from pyomo.core import Block, ConstraintList, Set, Constraint
from pyomo.core.base import Reference
from pyomo.common.modeling import unique_component_name
from pyomo.gdp.disjunct import Disjunct, Disjunction
import logging
def _clone_all_but_indicator_vars(self):
    """Clone everything in a Disjunct except for the indicator_vars"""
    return self.clone({id(self.indicator_var): self.indicator_var, id(self.binary_indicator_var): self.binary_indicator_var})