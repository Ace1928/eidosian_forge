import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo

        Reduce the rank of the matrix by retaining some SVD components.

        This corresponds to the "Broyden Rank Reduction Inverse"
        algorithm described in [1]_.

        Note that the SVD decomposition can be done by solving only a
        problem whose size is the effective rank of this matrix, which
        is viable even for large problems.

        Parameters
        ----------
        max_rank : int
            Maximum rank of this matrix after reduction.
        to_retain : int, optional
            Number of SVD components to retain when reduction is done
            (ie. rank > max_rank). Default is ``max_rank - 2``.

        References
        ----------
        .. [1] B.A. van der Rotten, PhD thesis,
           "A limited memory Broyden method to solve high-dimensional
           systems of nonlinear equations". Mathematisch Instituut,
           Universiteit Leiden, The Netherlands (2003).

           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

        