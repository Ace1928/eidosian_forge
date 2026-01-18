import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
class MaskedConstant(MaskedArray):
    __singleton = None

    @classmethod
    def __has_singleton(cls):
        return cls.__singleton is not None and type(cls.__singleton) is cls

    def __new__(cls):
        if not cls.__has_singleton():
            data = np.array(0.0)
            mask = np.array(True)
            data.flags.writeable = False
            mask.flags.writeable = False
            cls.__singleton = MaskedArray(data, mask=mask).view(cls)
        return cls.__singleton

    def __array_finalize__(self, obj):
        if not self.__has_singleton():
            return super().__array_finalize__(obj)
        elif self is self.__singleton:
            pass
        else:
            self.__class__ = MaskedArray
            MaskedArray.__array_finalize__(self, obj)

    def __array_prepare__(self, obj, context=None):
        return self.view(MaskedArray).__array_prepare__(obj, context)

    def __array_wrap__(self, obj, context=None):
        return self.view(MaskedArray).__array_wrap__(obj, context)

    def __str__(self):
        return str(masked_print_option._display)

    def __repr__(self):
        if self is MaskedConstant.__singleton:
            return 'masked'
        else:
            return object.__repr__(self)

    def __format__(self, format_spec):
        try:
            return object.__format__(self, format_spec)
        except TypeError:
            warnings.warn('Format strings passed to MaskedConstant are ignored, but in future may error or produce different behavior', FutureWarning, stacklevel=2)
            return object.__format__(self, '')

    def __reduce__(self):
        """Override of MaskedArray's __reduce__.
        """
        return (self.__class__, ())

    def __iop__(self, other):
        return self
    __iadd__ = __isub__ = __imul__ = __ifloordiv__ = __itruediv__ = __ipow__ = __iop__
    del __iop__

    def copy(self, *args, **kwargs):
        """ Copy is a no-op on the maskedconstant, as it is a scalar """
        return self

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __setattr__(self, attr, value):
        if not self.__has_singleton():
            return super().__setattr__(attr, value)
        elif self is self.__singleton:
            raise AttributeError(f'attributes of {self!r} are not writeable')
        else:
            return super().__setattr__(attr, value)