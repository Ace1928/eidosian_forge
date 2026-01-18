from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
class sos(ISOS):
    """A Special Ordered Set of type n."""
    _ctype = ISOS
    __slots__ = ('_parent', '_storage_key', '_active', '_variables', '_weights', '_level', '__weakref__')

    def __init__(self, variables, weights=None, level=1):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._variables = tuple(variables)
        self._weights = None
        self._level = level
        if weights is None:
            self._weights = tuple(range(1, len(self._variables) + 1))
        else:
            self._weights = tuple(weights)
            for w in self._weights:
                if not is_numeric_data(w):
                    raise ValueError('Weights for Special Ordered Sets must be expressions restricted to numeric data')
        assert len(self._variables) == len(self._weights)
        assert self._level >= 1

    @property
    def variables(self):
        return self._variables

    @property
    def weights(self):
        return self._weights

    @property
    def level(self):
        return self._level