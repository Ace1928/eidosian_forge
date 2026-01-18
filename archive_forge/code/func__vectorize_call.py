import collections.abc
import functools
import re
import sys
import warnings
from .._utils import set_module
import numpy as np
import numpy.core.numeric as _nx
from numpy.core import transpose
from numpy.core.numeric import (
from numpy.core.umath import (
from numpy.core.fromnumeric import (
from numpy.core.numerictypes import typecodes
from numpy.core import overrides
from numpy.core.function_base import add_newdoc
from numpy.lib.twodim_base import diag
from numpy.core.multiarray import (
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc
import builtins
from numpy.lib.histograms import histogram, histogramdd  # noqa: F401
def _vectorize_call(self, func, args):
    """Vectorized call to `func` over positional `args`."""
    if self.signature is not None:
        res = self._vectorize_call_with_signature(func, args)
    elif not args:
        res = func()
    else:
        ufunc, otypes = self._get_ufunc_and_otypes(func=func, args=args)
        inputs = [asanyarray(a, dtype=object) for a in args]
        outputs = ufunc(*inputs)
        if ufunc.nout == 1:
            res = asanyarray(outputs, dtype=otypes[0])
        else:
            res = tuple([asanyarray(x, dtype=t) for x, t in zip(outputs, otypes)])
    return res