import sys
import logging
from pyomo.common.deprecation import deprecated
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.expr.expr_common import _and, _or, _equiv, _inv, _xor, _impl
from pyomo.core.pyomoobject import PyomoObject
def _generate_logical_proposition(etype, _self, _other):
    raise RuntimeError('Incomplete import of Pyomo expression system')