import numpy as np
from scipy.linalg._basic import solve, solve_triangular
from scipy.sparse._base import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse._sputils import is_pydata_spmatrix, isintlike
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse._construct import eye
from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm
class ProductOperator(LinearOperator):
    """
    For now, this is limited to products of multiple square matrices.
    """

    def __init__(self, *args, **kwargs):
        self._structure = kwargs.get('structure', None)
        for A in args:
            if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
                raise ValueError('For now, the ProductOperator implementation is limited to the product of multiple square matrices.')
        if args:
            n = args[0].shape[0]
            for A in args:
                for d in A.shape:
                    if d != n:
                        raise ValueError('The square matrices of the ProductOperator must all have the same shape.')
            self.shape = (n, n)
            self.ndim = len(self.shape)
        self.dtype = np.result_type(*[x.dtype for x in args])
        self._operator_sequence = args

    def _matvec(self, x):
        for A in reversed(self._operator_sequence):
            x = A.dot(x)
        return x

    def _rmatvec(self, x):
        x = x.ravel()
        for A in self._operator_sequence:
            x = A.T.dot(x)
        return x

    def _matmat(self, X):
        for A in reversed(self._operator_sequence):
            X = _smart_matrix_product(A, X, structure=self._structure)
        return X

    @property
    def T(self):
        T_args = [A.T for A in reversed(self._operator_sequence)]
        return ProductOperator(*T_args)