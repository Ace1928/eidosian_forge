import functools
import types
import warnings
import numpy as np
from .._utils import set_module
from . import multiarray as mu
from . import overrides
from . import umath as um
from . import numerictypes as nt
from .multiarray import asarray, array, asanyarray, concatenate
from . import _methods
def _trace_dispatcher(a, offset=None, axis1=None, axis2=None, dtype=None, out=None):
    return (a, out)