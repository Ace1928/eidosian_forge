import numpy as np
from ._miobase import convert_dtypes
class MatlabObject(np.ndarray):
    """Subclass of ndarray to signal this is a matlab object.

    This is a simple subclass of :class:`numpy.ndarray` meant to be used
    by :func:`scipy.io.loadmat` and should not be instantiated directly.
    """

    def __new__(cls, input_array, classname=None):
        obj = np.asarray(input_array).view(cls)
        obj.classname = classname
        return obj

    def __array_finalize__(self, obj):
        self.classname = getattr(obj, 'classname', None)