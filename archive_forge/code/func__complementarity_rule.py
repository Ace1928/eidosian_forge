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
@staticmethod
def _complementarity_rule(b, *idx):
    _rule = b.parent_component()._init_rule
    if _rule is None:
        return
    cc = _rule(b.parent_block(), idx)
    if cc is None:
        raise ValueError('\nInvalid complementarity condition.  The complementarity condition\nis None instead of a 2-tuple.  Please modify your rule to return\nComplementarity.Skip instead of None.\n\nError thrown for Complementarity "%s".' % (b.name,))
    b.set_value(cc)