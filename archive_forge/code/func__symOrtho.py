import numpy
import cupy
import cupyx.cusolver
from cupy import cublas
from cupyx import cusparse
from cupy.cuda import cusolver
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import _interface
from cupyx.scipy.sparse.linalg._iterative import _make_system
import warnings
def _symOrtho(a, b):
    """
    A stable implementation of Givens rotation according to
    S.-C. Choi, "Iterative Methods for Singular Linear Equations
      and Least-Squares Problems", Dissertation,
      http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b == 0:
        return (numpy.sign(a), 0, abs(a))
    elif a == 0:
        return (0, numpy.sign(b), abs(b))
    elif abs(b) > abs(a):
        tau = a / b
        s = numpy.sign(b) / numpy.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = numpy.sign(a) / numpy.sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return (c, s, r)