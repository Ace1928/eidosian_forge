from pyomo.core.expr import ProductExpression, PowExpression
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core import Binary, value
from pyomo.core.base import (
from pyomo.core.base.var import _VarData
import logging
def _discretize_term(self, _expr, _x1, _x2, _block, _discretize, _known_bilinear):
    if id(_x1) in _discretize:
        _v = _x1
        _u = _x2
    elif id(_x2) in _discretize:
        _u = _x1
        _v = _x2
    else:
        raise RuntimeError("Couldn't identify discretized variable for expression '%s'!" % _expr)
    _id = (id(_v), id(_u))
    if _id not in _known_bilinear:
        _known_bilinear[_id] = self._discretize_bilinear(_block, _v, _discretize[id(_v)], _u, len(_known_bilinear))
    _expr._numerator = [_known_bilinear[_id]]