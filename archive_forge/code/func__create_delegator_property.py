from __future__ import annotations
from typing import (
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
def _create_delegator_property(name: str):

    def _getter(self):
        return self._delegate_property_get(name)

    def _setter(self, new_values):
        return self._delegate_property_set(name, new_values)
    _getter.__name__ = name
    _setter.__name__ = name
    return property(fget=_getter, fset=_setter, doc=getattr(delegate, accessor_mapping(name)).__doc__)