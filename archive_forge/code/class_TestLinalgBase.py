import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
class TestLinalgBase(EnableNRTStatsMixin, TestCase):
    """
    Provides setUp and common data/error modes for testing np.linalg functions.
    """
    dtypes = (np.float64, np.float32, np.complex128, np.complex64)

    def setUp(self):
        gc.collect()
        super(TestLinalgBase, self).setUp()

    def sample_vector(self, n, dtype):
        base = np.arange(n)
        if issubclass(dtype, np.complexfloating):
            return (base * (1 - 0.5j) + 2j).astype(dtype)
        else:
            return (base * 0.5 + 1).astype(dtype)

    def specific_sample_matrix(self, size, dtype, order, rank=None, condition=None):
        """
        Provides a sample matrix with an optionally specified rank or condition
        number.

        size: (rows, columns), the dimensions of the returned matrix.
        dtype: the dtype for the returned matrix.
        order: the memory layout for the returned matrix, 'F' or 'C'.
        rank: the rank of the matrix, an integer value, defaults to full rank.
        condition: the condition number of the matrix (defaults to 1.)

        NOTE: Only one of rank or condition may be set.
        """
        d_cond = 1.0
        if len(size) != 2:
            raise ValueError('size must be a length 2 tuple.')
        if order not in ['F', 'C']:
            raise ValueError("order must be one of 'F' or 'C'.")
        if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
            raise ValueError('dtype must be a numpy floating point type.')
        if rank is not None and condition is not None:
            raise ValueError('Only one of rank or condition can be specified.')
        if condition is None:
            condition = d_cond
        if condition < 1:
            raise ValueError('Condition number must be >=1.')
        np.random.seed(0)
        m, n = size
        if m < 0 or n < 0:
            raise ValueError('Negative dimensions given for matrix shape.')
        minmn = min(m, n)
        if rank is None:
            rv = minmn
        else:
            if rank <= 0:
                raise ValueError('Rank must be greater than zero.')
            if not isinstance(rank, Integral):
                raise ValueError('Rank must an integer.')
            rv = rank
            if rank > minmn:
                raise ValueError('Rank given greater than full rank.')
        if m == 1 or n == 1:
            if condition != d_cond:
                raise ValueError('Condition number was specified for a vector (always 1.).')
            maxmn = max(m, n)
            Q = self.sample_vector(maxmn, dtype).reshape(m, n)
        else:
            tmp = self.sample_vector(m * m, dtype).reshape(m, m)
            U, _ = np.linalg.qr(tmp)
            tmp = self.sample_vector(n * n, dtype)[::-1].reshape(n, n)
            V, _ = np.linalg.qr(tmp)
            sv = np.linspace(d_cond, condition, rv)
            S = np.zeros((m, n))
            idx = np.nonzero(np.eye(m, n))
            S[idx[0][:rv], idx[1][:rv]] = sv
            Q = np.dot(np.dot(U, S), V.T)
            Q = np.array(Q, dtype=dtype, order=order)
        return Q

    def assert_error(self, cfunc, args, msg, err=ValueError):
        with self.assertRaises(err) as raises:
            cfunc(*args)
        self.assertIn(msg, str(raises.exception))

    def assert_non_square(self, cfunc, args):
        msg = 'Last 2 dimensions of the array must be square.'
        self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)

    def assert_wrong_dtype(self, name, cfunc, args):
        msg = 'np.linalg.%s() only supported on float and complex arrays' % name
        self.assert_error(cfunc, args, msg, errors.TypingError)

    def assert_wrong_dimensions(self, name, cfunc, args, la_prefix=True):
        prefix = 'np.linalg' if la_prefix else 'np'
        msg = '%s.%s() only supported on 2-D arrays' % (prefix, name)
        self.assert_error(cfunc, args, msg, errors.TypingError)

    def assert_no_nan_or_inf(self, cfunc, args):
        msg = 'Array must not contain infs or NaNs.'
        self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)

    def assert_contig_sanity(self, got, expected_contig):
        """
        This checks that in a computed result from numba (array, possibly tuple
        of arrays) all the arrays are contiguous in memory and that they are
        all at least one of "C_CONTIGUOUS" or "F_CONTIGUOUS". The computed
        result of the contiguousness is then compared against a hardcoded
        expected result.

        got: is the computed results from numba
        expected_contig: is "C" or "F" and is the expected type of
                        contiguousness across all input values
                        (and therefore tests).
        """
        if isinstance(got, tuple):
            for a in got:
                self.assert_contig_sanity(a, expected_contig)
        elif not isinstance(got, Number):
            c_contig = got.flags.c_contiguous
            f_contig = got.flags.f_contiguous
            msg = 'Results are not at least one of all C or F contiguous.'
            self.assertTrue(c_contig | f_contig, msg)
            msg = 'Computed contiguousness does not match expected.'
            if expected_contig == 'C':
                self.assertTrue(c_contig, msg)
            elif expected_contig == 'F':
                self.assertTrue(f_contig, msg)
            else:
                raise ValueError('Unknown contig')

    def assert_raise_on_singular(self, cfunc, args):
        msg = 'Matrix is singular to machine precision.'
        self.assert_error(cfunc, args, msg, err=np.linalg.LinAlgError)

    def assert_is_identity_matrix(self, got, rtol=None, atol=None):
        """
        Checks if a matrix is equal to the identity matrix.
        """
        self.assertEqual(got.shape[-1], got.shape[-2])
        eye = np.eye(got.shape[-1], dtype=got.dtype)
        resolution = 5 * np.finfo(got.dtype).resolution
        if rtol is None:
            rtol = 10 * resolution
        if atol is None:
            atol = 100 * resolution
        np.testing.assert_allclose(got, eye, rtol, atol)

    def assert_invalid_norm_kind(self, cfunc, args):
        """
        For use in norm() and cond() tests.
        """
        msg = 'Invalid norm order for matrices.'
        self.assert_error(cfunc, args, msg, ValueError)

    def assert_raise_on_empty(self, cfunc, args):
        msg = 'Arrays cannot be empty'
        self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)