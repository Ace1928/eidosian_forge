import atexit
import binascii
import functools
import hashlib
import operator
import os
import time
import numpy
import warnings
from numpy.linalg import LinAlgError
import cupy
from cupy import _core
from cupy import cuda
from cupy.cuda import curand
from cupy.cuda import device
from cupy.random import _kernels
from cupy import _util
import cupyx
def _interval(self, mx, size):
    """Generate multiple integers independently sampled uniformly from ``[0, mx]``.

        Args:
            mx (int): Upper bound of the interval
            size (None or int or tuple): Shape of the array or the scalar
                returned.
        Returns:
            int or cupy.ndarray: If ``None``, an :class:`cupy.ndarray` with
            shape ``()`` is returned.
            If ``int``, 1-D array of length size is returned.
            If ``tuple``, multi-dimensional array with shape
            ``size`` is returned.
            Currently, only 32 bit or 64 bit integers can be sampled.
        """
    if size is None:
        size = ()
    elif isinstance(size, int):
        size = (size,)
    is_mx_scalar = numpy.isscalar(mx)
    if is_mx_scalar:
        if mx == 0:
            return cupy.zeros(size, dtype=numpy.uint32)
        if mx < 0:
            raise ValueError('mx must be non-negative (actual: {})'.format(mx))
        elif mx <= _UINT32_MAX:
            dtype = numpy.uint32
            upper_limit = _UINT32_MAX - (1 << 32) % (mx + 1)
        elif mx <= _UINT64_MAX:
            dtype = numpy.uint64
            upper_limit = _UINT64_MAX - (1 << 64) % (mx + 1)
        else:
            raise ValueError('mx must be within uint64 range (actual: {})'.format(mx))
    else:
        dtype = mx.dtype
        if dtype == cupy.int32 or dtype == cupy.uint32:
            dtype = numpy.uint32
            mx = mx.astype(dtype, copy=False)
            upper_limit = self._interval_upper_limit(_UINT32_MAX, mx)
        elif dtype == cupy.int64 or dtype == cupy.uint64:
            dtype = numpy.uint64
            mx = mx.astype(dtype, copy=False)
            upper_limit = self._interval_upper_limit(_UINT64_MAX, mx)
        else:
            raise ValueError('dtype must be integer, got: {}'.format(dtype))
    n_sample = functools.reduce(operator.mul, size, 1)
    if n_sample == 0:
        return cupy.empty(size, dtype=dtype)
    sample = self._curand_generate(n_sample, dtype)
    if is_mx_scalar:
        mx1 = mx + 1
        if mx1 == 1 << mx1.bit_length() - 1:
            mask = (1 << mx.bit_length()) - 1
            sample &= mask
            return sample.reshape(size)
    ng_indices = self._get_indices(sample, upper_limit, False)
    n_ng = ng_indices.size
    if n_ng > 0 and (not numpy.isscalar(mx)):
        upper_limit = upper_limit[ng_indices]
    while n_ng > 0:
        n_supplement = max(n_ng * 2, 1024) if is_mx_scalar else upper_limit.size
        supplement = self._curand_generate(n_supplement, dtype)
        ok_indices = self._get_indices(supplement, upper_limit, True)
        n_ok = ok_indices.size
        if n_ok >= n_ng:
            sample[ng_indices] = supplement[ok_indices[:n_ng]]
            n_ng = 0
        else:
            sample[ng_indices[:n_ok]] = supplement[ok_indices]
            ng_indices = ng_indices[n_ok:]
            if not is_mx_scalar:
                upper_limit = upper_limit[n_ok:]
            n_ng -= n_ok
    if is_mx_scalar:
        sample %= mx1
    elif dtype == cupy.uint32:
        sample = self._interval_sample_modulo(_UINT32_MAX, mx, sample)
    else:
        sample = self._interval_sample_modulo(_UINT64_MAX, mx, sample)
    return sample.reshape(size)