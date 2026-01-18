import logging
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.core import Block, Constraint, Var
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction
def _activated_disjuncts_in_disjunction(disjtn):
    """Retrieve generator of activated disjuncts on disjunction."""
    return (disj for disj in disjtn.disjuncts if disj.active and (not disj.indicator_var.fixed))