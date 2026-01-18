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
def _raise_disjunct_in_multiple_disjunctions_error(disjunct, disjunction):
    raise GDP_Error("The disjunct '%s' has been transformed, but '%s', a disjunction it appears in, has not. Putting the same disjunct in multiple disjunctions is not supported." % (disjunct.name, disjunction.name))