from __future__ import annotations
from functools import partial
from operator import getitem
import numpy as np
from dask import core
from dask.array.core import Array, apply_infer_dtype, asarray, blockwise, elemwise
from dask.base import is_dask_collection, normalize_token
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from, funcname
class ufunc:
    _forward_attrs = {'nin', 'nargs', 'nout', 'ntypes', 'identity', 'signature', 'types'}

    def __init__(self, ufunc):
        if not isinstance(ufunc, (np.ufunc, da_frompyfunc)):
            raise TypeError('must be an instance of `ufunc` or `da_frompyfunc`, got `%s' % type(ufunc).__name__)
        self._ufunc = ufunc
        self.__name__ = ufunc.__name__
        if isinstance(ufunc, np.ufunc):
            derived_from(np)(self)

    def __dask_tokenize__(self):
        return (self.__name__, normalize_token(self._ufunc))

    def __getattr__(self, key):
        if key in self._forward_attrs:
            return getattr(self._ufunc, key)
        raise AttributeError(f'{type(self).__name__!r} object has no attribute {key!r}')

    def __dir__(self):
        return list(self._forward_attrs.union(dir(type(self)), self.__dict__))

    def __repr__(self):
        return repr(self._ufunc)

    def __call__(self, *args, **kwargs):
        dsks = [arg for arg in args if hasattr(arg, '_elemwise')]
        if len(dsks) > 0:
            for dsk in dsks:
                result = dsk._elemwise(self._ufunc, *args, **kwargs)
                if type(result) != type(NotImplemented):
                    return result
            raise TypeError('Parameters of such types are not supported by ' + self.__name__)
        else:
            return self._ufunc(*args, **kwargs)

    @derived_from(np.ufunc)
    def outer(self, A, B, **kwargs):
        if self.nin != 2:
            raise ValueError('outer product only supported for binary functions')
        if 'out' in kwargs:
            raise ValueError('`out` kwarg not supported')
        A_is_dask = is_dask_collection(A)
        B_is_dask = is_dask_collection(B)
        if not A_is_dask and (not B_is_dask):
            return self._ufunc.outer(A, B, **kwargs)
        elif A_is_dask and (not isinstance(A, Array)) or (B_is_dask and (not isinstance(B, Array))):
            raise NotImplementedError('Dask objects besides `dask.array.Array` are not supported at this time.')
        A = asarray(A)
        B = asarray(B)
        ndim = A.ndim + B.ndim
        out_inds = tuple(range(ndim))
        A_inds = out_inds[:A.ndim]
        B_inds = out_inds[A.ndim:]
        dtype = apply_infer_dtype(self._ufunc.outer, [A, B], kwargs, 'ufunc.outer', suggest_dtype=False)
        if 'dtype' in kwargs:
            func = partial(self._ufunc.outer, dtype=kwargs.pop('dtype'))
        else:
            func = self._ufunc.outer
        return blockwise(func, out_inds, A, A_inds, B, B_inds, dtype=dtype, token=self.__name__ + '.outer', **kwargs)