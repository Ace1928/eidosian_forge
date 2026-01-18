from contextlib import contextmanager
import numpy as np
from_record_like = None
class FakeWithinKernelCUDAArray(object):
    """
    Created to emulate the behavior of arrays within kernels, where either
    array.item or array['item'] is valid (that is, give all structured
    arrays `numpy.recarray`-like semantics). This behaviour does not follow
    the semantics of Python and NumPy with non-jitted code, and will be
    deprecated and removed.
    """

    def __init__(self, item):
        assert isinstance(item, FakeCUDAArray)
        self.__dict__['_item'] = item

    def __wrap_if_fake(self, item):
        if isinstance(item, FakeCUDAArray):
            return FakeWithinKernelCUDAArray(item)
        else:
            return item

    def __getattr__(self, attrname):
        if attrname in dir(self._item._ary):
            return self.__wrap_if_fake(getattr(self._item._ary, attrname))
        else:
            return self.__wrap_if_fake(self._item.__getitem__(attrname))

    def __setattr__(self, nm, val):
        self._item.__setitem__(nm, val)

    def __getitem__(self, idx):
        return self.__wrap_if_fake(self._item.__getitem__(idx))

    def __setitem__(self, idx, val):
        self._item.__setitem__(idx, val)

    def __len__(self):
        return len(self._item)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        call = getattr(ufunc, method)

        def convert_fakes(obj):
            if isinstance(obj, FakeWithinKernelCUDAArray):
                obj = obj._item._ary
            return obj
        out = kwargs.get('out')
        if out:
            kwargs['out'] = tuple((convert_fakes(o) for o in out))
        args = tuple((convert_fakes(a) for a in args))
        return call(*args, **kwargs)