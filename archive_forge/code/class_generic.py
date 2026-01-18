import builtins
import torch
from . import _dtypes_impl
class generic:

    @property
    def name(self):
        return self.__class__.__name__

    def __new__(cls, value):
        from ._ndarray import asarray, ndarray
        if isinstance(value, str) and value in ['inf', 'nan']:
            value = {'inf': torch.inf, 'nan': torch.nan}[value]
        if isinstance(value, ndarray):
            return value.astype(cls)
        else:
            return asarray(value, dtype=cls)