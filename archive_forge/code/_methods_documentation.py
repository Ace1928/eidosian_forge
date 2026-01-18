import warnings
from contextlib import nullcontext
from numpy.core import multiarray as mu
from numpy.core import umath as um
from numpy.core.multiarray import asanyarray
from numpy.core import numerictypes as nt
from numpy.core import _exceptions
from numpy.core._ufunc_config import _no_nep50_warning
from numpy._globals import _NoValue
from numpy.compat import pickle, os_fspath

Array methods which are called by both the C-code for the method
and the Python code for the NumPy-namespace function

