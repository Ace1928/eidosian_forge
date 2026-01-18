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
def _find_parent_disjunct(constraint):
    parent_disjunct = constraint.parent_block()
    while not isinstance(parent_disjunct, _DisjunctData):
        if parent_disjunct is None:
            raise GDP_Error("Constraint '%s' is not on a disjunct and so was not transformed" % constraint.name)
        parent_disjunct = parent_disjunct.parent_block()
    return parent_disjunct