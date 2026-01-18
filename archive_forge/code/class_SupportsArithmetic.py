from __future__ import annotations
import numbers
import numpy as np
from xarray.core._typed_ops import (
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.ops import IncludeNumpySameMethods, IncludeReduceMethods
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.namedarray.utils import is_duck_array
class SupportsArithmetic:
    """Base class for xarray types that support arithmetic.

    Used by Dataset, DataArray, Variable and GroupBy.
    """
    __slots__ = ()
    _HANDLED_TYPES = (np.generic, numbers.Number, bytes, str)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        from xarray.core.computation import apply_ufunc
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not is_duck_array(x) and (not isinstance(x, self._HANDLED_TYPES + (SupportsArithmetic,))):
                return NotImplemented
        if ufunc.signature is not None:
            raise NotImplementedError(f'{ufunc} not supported: xarray objects do not directly implement generalized ufuncs. Instead, use xarray.apply_ufunc or explicitly convert to xarray objects to NumPy arrays (e.g., with `.values`).')
        if method != '__call__':
            raise NotImplementedError('{} method for ufunc {} is not implemented on xarray objects, which currently only support the __call__ method. As an alternative, consider explicitly converting xarray objects to NumPy arrays (e.g., with `.values`).'.format(method, ufunc))
        if any((isinstance(o, SupportsArithmetic) for o in out)):
            raise NotImplementedError('xarray objects are not yet supported in the `out` argument for ufuncs. As an alternative, consider explicitly converting xarray objects to NumPy arrays (e.g., with `.values`).')
        join = dataset_join = OPTIONS['arithmetic_join']
        return apply_ufunc(ufunc, *inputs, input_core_dims=((),) * ufunc.nin, output_core_dims=((),) * ufunc.nout, join=join, dataset_join=dataset_join, dataset_fill_value=np.nan, kwargs=kwargs, dask='allowed', keep_attrs=_get_keep_attrs(default=True))