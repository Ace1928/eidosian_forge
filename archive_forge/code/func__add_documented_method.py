from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.linalg import LinAlgError
from scipy import special, stats
from . import _remove_indexes_to_reduce
from .linalg import cholesky, eigh
def _add_documented_method(cls, wrapped_cls, methods, extra_docs=None):
    """Register methods to XrRV classes and document them from a template."""
    if extra_docs is None:
        extra_docs = {}
    for method_name in methods:
        extra_doc = extra_docs.get(method_name, '')
        if method_name == 'rvs':
            if wrapped_cls == 'rv_generic':
                continue
            method = cls.rvs
        else:
            method = _wrap_method(method_name)
        setattr(method, '__doc__', f'Method wrapping :meth:`scipy.stats.{wrapped_cls}.{method_name}` with :func:`xarray.apply_ufunc`\n\nUsage examples available at :ref:`stats_tutorial/dists`.\n\n{extra_doc}')
        setattr(cls, method_name, method)