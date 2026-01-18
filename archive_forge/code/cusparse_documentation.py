import functools as _functools
import numpy as _numpy
import platform as _platform
import cupy as _cupy
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupy import _util
import cupyx.scipy.sparse
Matrix-matrix product for CSR-matrix.

    math::
       C = alpha * A * B

    Args:
        a (cupyx.scipy.sparse.csr_matrix): Sparse matrix A.
        b (cupyx.scipy.sparse.csr_matrix): Sparse matrix B.
        alpha (scalar): Coefficient

    Returns:
        cupyx.scipy.sparse.csr_matrix

    