import functools
import itertools
import operator
import sys
import warnings
import numbers
import builtins
import numpy as np
from . import multiarray
from .multiarray import (
from . import overrides
from . import umath
from . import shape_base
from .overrides import set_array_function_like_doc, set_module
from .umath import (multiply, invert, sin, PINF, NAN)
from . import numerictypes
from .numerictypes import longlong, intc, int_, float_, complex_, bool_
from ..exceptions import ComplexWarning, TooHardError, AxisError
from ._ufunc_config import errstate, _no_nep50_warning
from .umath import *
from .numerictypes import *
from . import fromnumeric
from .fromnumeric import *
from . import arrayprint
from .arrayprint import *
from . import _asarray
from ._asarray import *
from . import _ufunc_config
from ._ufunc_config import *
def _convolve_dispatcher(a, v, mode=None):
    return (a, v)