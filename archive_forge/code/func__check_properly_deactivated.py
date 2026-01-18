from pyomo.gdp import GDP_Error, Disjunction
from pyomo.gdp.disjunct import _DisjunctData, Disjunct
import pyomo.core.expr as EXPR
from pyomo.core.base.component import _ComponentBase
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentMap, ComponentSet, OrderedSet
from pyomo.opt import TerminationCondition, SolverStatus
from weakref import ref as weakref_ref
from collections import defaultdict
import logging
def _check_properly_deactivated(disjunct):
    if disjunct.indicator_var.is_fixed():
        if not value(disjunct.indicator_var):
            return
        else:
            raise GDP_Error("The disjunct '%s' is deactivated, but the indicator_var is fixed to %s. This makes no sense." % (disjunct.name, value(disjunct.indicator_var)))
    if disjunct._transformation_block is None:
        raise GDP_Error("The disjunct '%s' is deactivated, but the indicator_var is not fixed and the disjunct does not appear to have been transformed. This makes no sense. (If the intent is to deactivate the disjunct, fix its indicator_var to False.)" % (disjunct.name,))