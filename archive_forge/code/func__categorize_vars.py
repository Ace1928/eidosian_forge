import ctypes
import logging
import os
from collections import deque, defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter, setitem
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, MouseTrap
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import (
from pyomo.common.timing import TicTocTimer
from pyomo.core.expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import (
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import _VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory
from pyomo.repn.util import (
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port
def _categorize_vars(self, comp_list, linear_by_comp):
    """Categorize compiled expression vars into linear and nonlinear

        This routine takes an iterable of compiled component expression
        infos and returns the sets of variables appearing linearly and
        nonlinearly in those components.

        This routine has a number of side effects:

          - the ``linear_by_comp`` dict is updated to contain the set of
            nonzeros for each component in the ``comp_list``

          - the expr_info (the second element in each tuple in
            ``comp_list``) is "compiled": the ``linear`` attribute is
            converted from a list of coef, var_id terms (potentially with
            duplicate entries) into a dict that maps var id to
            coefficients

        Returns
        -------
        all_linear_vars: set
            set of all vars that only appear linearly in the compiled
            component expression infos

        all_nonlinear_vars: set
            set of all vars that appear nonlinearly in the compiled
            component expression infos

        nnz_by_var: dict
            Count of the number of components that each var appears in.

        """
    all_linear_vars = set()
    all_nonlinear_vars = set()
    nnz_by_var = {}
    for comp_info in comp_list:
        expr_info = comp_info[1]
        if expr_info.linear:
            linear_vars = set(expr_info.linear)
            all_linear_vars.update(linear_vars)
        if expr_info.nonlinear:
            nonlinear_vars = set()
            for _id in expr_info.nonlinear[1]:
                if _id in nonlinear_vars:
                    continue
                if _id in linear_by_comp:
                    nonlinear_vars.update(linear_by_comp[_id])
                else:
                    nonlinear_vars.add(_id)
            if expr_info.linear:
                for i in filterfalse(linear_vars.__contains__, nonlinear_vars):
                    expr_info.linear[i] = 0
            else:
                expr_info.linear = dict.fromkeys(nonlinear_vars, 0)
            all_nonlinear_vars.update(nonlinear_vars)
        for v in expr_info.linear:
            if v in nnz_by_var:
                nnz_by_var[v] += 1
            else:
                nnz_by_var[v] = 1
        linear_by_comp[id(comp_info[0])] = expr_info.linear
    if all_nonlinear_vars:
        all_linear_vars -= all_nonlinear_vars
    return (all_linear_vars, all_nonlinear_vars, nnz_by_var)