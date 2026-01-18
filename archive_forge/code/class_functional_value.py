from pyomo.core.expr.numvalue import is_numeric_data, NumericValue
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.container_utils import define_simple_containers
class functional_value(IParameter):
    """An object for storing a numeric function that can be
    used in a symbolic expression.

    Note that models making use of this object may require
    the dill module for serialization.
    """
    _ctype = IParameter
    __slots__ = ('_parent', '_storage_key', '_active', '_fn', '__weakref__')

    def __init__(self, fn=None):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._fn = fn

    def __call__(self, exception=True):
        if self._fn is None:
            return None
        try:
            val = self._fn()
        except Exception as e:
            if exception:
                raise e
            else:
                return None
        if not is_numeric_data(val):
            raise TypeError('Functional value is not numeric data')
        return val

    @property
    def fn(self):
        """The function stored with this object"""
        return self._fn

    @fn.setter
    def fn(self, fn):
        self._fn = fn