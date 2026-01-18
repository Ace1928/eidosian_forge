from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import NOTSET
import pyomo.core.expr as EXPR
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.expr.numvalue import (
class expression(IExpression):
    """A named, mutable expression."""
    _ctype = IExpression
    __slots__ = ('_parent', '_storage_key', '_active', '_expr', '__weakref__')

    def __init__(self, expr=None):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._expr = None
        self.expr = expr

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, expr):
        self._expr = expr