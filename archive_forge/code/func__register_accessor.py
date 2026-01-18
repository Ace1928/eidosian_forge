from __future__ import annotations
import warnings
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
def _register_accessor(name, cls):

    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(f'registration of accessor {accessor!r} under name {name!r} for type {cls!r} is overriding a preexisting attribute with the same name.', AccessorRegistrationWarning, stacklevel=2)
        setattr(cls, name, _CachedAccessor(name, accessor))
        return accessor
    return decorator