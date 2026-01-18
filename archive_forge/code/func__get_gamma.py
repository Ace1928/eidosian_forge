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
def _get_gamma(virtual_indexes, previous_indexes, method):
    """
    Compute gamma (a.k.a 'm' or 'weight') for the linear interpolation
    of quantiles.

    virtual_indexes : array_like
        The indexes where the percentile is supposed to be found in the sorted
        sample.
    previous_indexes : array_like
        The floor values of virtual_indexes.
    interpolation : dict
        The interpolation method chosen, which may have a specific rule
        modifying gamma.

    gamma is usually the fractional part of virtual_indexes but can be modified
    by the interpolation method.
    """
    gamma = np.asanyarray(virtual_indexes - previous_indexes)
    gamma = method['fix_gamma'](gamma, virtual_indexes)
    return np.asanyarray(gamma)