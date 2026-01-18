import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
class ClassWithAny(HasTraits):
    x = Property
    _x = Any

    def _get_x(self):
        return self._x

    def _set_x(self, x):
        self._x = x