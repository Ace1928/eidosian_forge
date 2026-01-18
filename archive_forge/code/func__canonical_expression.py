from collections import namedtuple
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import ZeroConstant, native_numeric_types, as_numeric
from pyomo.core import Constraint, Var, Block, Set
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.block import _BlockData
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
import logging
def _canonical_expression(self, e):
    e_ = None
    if e.__class__ is EXPR.EqualityExpression:
        if e.arg(1).__class__ in native_numeric_types or e.arg(1).is_fixed():
            _e = (e.arg(1), e.arg(0))
        else:
            _e = (ZeroConstant, e.arg(0) - e.arg(1))
    elif e.__class__ is EXPR.InequalityExpression:
        if e.arg(1).__class__ in native_numeric_types or e.arg(1).is_fixed():
            _e = (None, e.arg(0), e.arg(1))
        elif e.arg(0).__class__ in native_numeric_types or e.arg(0).is_fixed():
            _e = (e.arg(0), e.arg(1), None)
        else:
            _e = (ZeroConstant, e.arg(1) - e.arg(0), None)
    elif e.__class__ is EXPR.RangedExpression:
        _e = (e.arg(0), e.arg(1), e.arg(2))
    else:
        _e = (None, e, None)
    return _e