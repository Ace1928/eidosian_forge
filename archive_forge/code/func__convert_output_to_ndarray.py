import functools
import inspect
import os
import random
from typing import Tuple, Type
import traceback
import unittest
import warnings
import numpy
import cupy
from cupy.testing import _array
from cupy.testing import _parameterized
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def _convert_output_to_ndarray(c_out, n_out, sp_name, check_sparse_format):
    """Checks type of cupy/numpy results and returns cupy/numpy ndarrays.

    Args:
        c_out (cupy.ndarray, cupyx.scipy.sparse matrix, cupy.poly1d or scalar):
            cupy result
        n_out (numpy.ndarray, scipy.sparse matrix, numpy.poly1d or scalar):
            numpy result
        sp_name(str or None): Argument name whose value is either
            ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
            argument is given for the modules.
        check_sparse_format (bool): If ``True``, consistency of format of
            sparse matrix is also checked. Default is ``True``.

    Returns:
        The tuple of cupy.ndarray and numpy.ndarray.
    """
    if sp_name is not None and cupyx.scipy.sparse.issparse(c_out):
        import scipy.sparse
        assert scipy.sparse.issparse(n_out)
        if check_sparse_format:
            assert c_out.format == n_out.format
        return (c_out.A, n_out.A)
    if isinstance(c_out, cupy.ndarray) and isinstance(n_out, (numpy.ndarray, numpy.generic)):
        return (c_out, n_out)
    if isinstance(c_out, cupy.poly1d) and isinstance(n_out, numpy.poly1d):
        assert c_out.variable == n_out.variable
        return (c_out.coeffs, n_out.coeffs)
    if isinstance(c_out, numpy.generic) and isinstance(n_out, numpy.generic):
        return (c_out, n_out)
    if numpy.isscalar(c_out) and numpy.isscalar(n_out):
        return (cupy.array(c_out), numpy.array(n_out))
    raise AssertionError('numpy and cupy returns different type of return value:\ncupy: {}\nnumpy: {}'.format(type(c_out), type(n_out)))