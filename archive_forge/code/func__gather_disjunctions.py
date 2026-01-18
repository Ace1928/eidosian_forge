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
def _gather_disjunctions(block, gdp_tree, include_root=True):
    if not include_root:
        root = block
    to_explore = [block]
    while to_explore:
        block = to_explore.pop()
        for disjunction in block.component_data_objects(Disjunction, active=True, sort=SortComponents.deterministic, descend_into=Block):
            gdp_tree.add_node(disjunction)
            for disjunct in disjunction.disjuncts:
                if not disjunct.active:
                    if disjunct.transformation_block is not None:
                        _raise_disjunct_in_multiple_disjunctions_error(disjunct, disjunction)
                    _check_properly_deactivated(disjunct)
                    continue
                gdp_tree.add_edge(disjunction, disjunct)
                to_explore.append(disjunct)
            if block.ctype is Disjunct:
                if not include_root and block is root:
                    continue
                gdp_tree.add_edge(block, disjunction)
    return gdp_tree