from .base import (
from scipy.sparse.linalg import splu, LinearOperator
from scipy.linalg import eigvals
from scipy.sparse import isspmatrix_csc, spmatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
import numpy as np
from typing import Union, Tuple, Optional, Callable
class ScipyIterative(LinearSolverInterface):

    def __init__(self, method: Callable, options=None):
        self.method = method
        if options is None:
            self.options = dict()
        else:
            self.options = dict(options)

    def solve(self, matrix: Union[spmatrix, BlockMatrix], rhs: Union[np.ndarray, BlockVector], raise_on_error: bool=True) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        linear_operator = _LinearOperator(matrix.tocoo())
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
        else:
            _rhs = rhs
        result, info = self.method(linear_operator, _rhs, **self.options)
        if info == 0:
            stat = LinearSolverStatus.successful
        else:
            stat = LinearSolverStatus.error
        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result
        return (result, LinearSolverResults(stat))