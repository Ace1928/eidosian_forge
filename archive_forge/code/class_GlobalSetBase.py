from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.range import NonNumericRange
class GlobalSetBase(PyomoObject):
    """The base class for all Global sets"""
    __slots__ = ()

    def __reduce__(self):
        return (_get_global_set, (self.local_name,))

    def __deepcopy__(self, memo):
        return self

    def __str__(self):
        return self.name

    @property
    def _parent(self):
        return None

    @_parent.setter
    def _parent(self, val):
        if val is None:
            return
        val = val()
        raise RuntimeError("Cannot assign a GlobalSet '%s' to %s '%s'" % (self.global_name, 'model' if val.model() is val else 'block', val.name or 'unknown'))