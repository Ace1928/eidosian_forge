from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.kernel.base import _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.kernel.expression import IExpression
class objective(IObjective):
    """An optimization objective."""
    _ctype = IObjective
    __slots__ = ('_parent', '_storage_key', '_active', '_expr', '_sense', '__weakref__')

    def __init__(self, expr=None, sense=minimize):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._expr = None
        self._sense = None
        self.sense = sense
        self.expr = expr

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, expr):
        self._expr = as_numeric(expr) if expr is not None else None

    @property
    def sense(self):
        return self._sense

    @sense.setter
    def sense(self, sense):
        """Set the sense (direction) of this objective."""
        if sense == minimize or sense == maximize:
            self._sense = sense
        else:
            raise ValueError("Objective sense must be set to one of: [minimize (%s), maximize (%s)]. Invalid value: %s'" % (minimize, maximize, sense))