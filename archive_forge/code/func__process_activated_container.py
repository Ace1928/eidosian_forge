import logging
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.core import Block, Constraint, Var
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction
def _process_activated_container(blk):
    """Process a container object, returning the new components found."""
    new_fixed_true_disjuncts = ComponentSet((disj for disj in blk.component_data_objects(Disjunct, active=True) if disj.indicator_var.value and disj.indicator_var.fixed))
    new_activated_disjunctions = ComponentSet(blk.component_data_objects(Disjunction, active=True))
    new_activated_disjuncts = ComponentSet((disj for disjtn in new_activated_disjunctions for disj in _activated_disjuncts_in_disjunction(disjtn)))
    new_activated_constraints = ComponentSet(blk.component_data_objects(Constraint, active=True))
    return (new_activated_disjunctions, new_fixed_true_disjuncts, new_activated_disjuncts, new_activated_constraints)