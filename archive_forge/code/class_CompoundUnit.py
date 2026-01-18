import weakref
import numpy
from .dimensionality import Dimensionality
from . import markup
from .quantity import Quantity, get_conversion_factor
from .registry import unit_registry
from .decorators import memoize, with_doc
class CompoundUnit(UnitQuantity):
    _primary_order = 99

    def __new__(cls, name):
        return UnitQuantity.__new__(cls, name, unit_registry[name])

    def __init__(self, name):
        return

    @with_doc(UnitQuantity.__add__, use_header=False)
    def __repr__(self):
        return '1 %s' % self.name

    @property
    def name(self):
        if markup.config.use_unicode:
            return '(%s)' % markup.superscript(self._name)
        else:
            return '(%s)' % self._name

    def __reduce__(self):
        """
        Return a tuple for pickling a UnitQuantity.
        """
        return (type(self), (self._name,), self.__getstate__())

    def copy(self):
        return type(self)(self._name)