import functools
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy._creation import from_data
from cupy._manipulation import join
class AxisConcatenator(object):
    """Translates slice objects to concatenation along an axis.

    For detailed documentation on usage, see :func:`cupy.r_`.
    This implementation is partially borrowed from NumPy's one.

    """

    def _output_obj(self, obj, ndim, ndmin, trans1d):
        k2 = ndmin - ndim
        if trans1d < 0:
            trans1d += k2 + 1
        defaxes = list(range(ndmin))
        k1 = trans1d
        axes = defaxes[:k1] + defaxes[k2:] + defaxes[k1:k2]
        return obj.transpose(axes)

    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        self.axis = axis
        self.trans1d = trans1d
        self.matrix = matrix
        self.ndmin = ndmin

    def __getitem__(self, key):
        trans1d = self.trans1d
        ndmin = self.ndmin
        objs = []
        scalars = []
        arraytypes = []
        scalartypes = []
        if isinstance(key, str):
            raise NotImplementedError
        if not isinstance(key, tuple):
            key = (key,)
        for i, k in enumerate(key):
            scalar = False
            if isinstance(k, slice):
                raise NotImplementedError
            elif isinstance(k, str):
                if i != 0:
                    raise ValueError('special directives must be the first entry.')
                raise NotImplementedError
            elif type(k) in numpy.ScalarType:
                newobj = from_data.array(k, ndmin=ndmin)
                scalars.append(i)
                scalar = True
                scalartypes.append(newobj.dtype)
            else:
                newobj = from_data.array(k, copy=False, ndmin=ndmin)
                if ndmin > 1:
                    ndim = from_data.array(k, copy=False).ndim
                    if trans1d != -1 and ndim < ndmin:
                        newobj = self._output_obj(newobj, ndim, ndmin, trans1d)
            objs.append(newobj)
            if not scalar and isinstance(newobj, _core.ndarray):
                arraytypes.append(newobj.dtype)
        final_dtype = numpy.find_common_type(arraytypes, scalartypes)
        if final_dtype is not None:
            for k in scalars:
                objs[k] = objs[k].astype(final_dtype)
        return join.concatenate(tuple(objs), axis=self.axis)

    def __len__(self):
        return 0