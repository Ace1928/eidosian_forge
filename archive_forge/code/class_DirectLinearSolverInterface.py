from abc import ABCMeta, abstractmethod
import enum
from typing import Optional, Union, Tuple
from scipy.sparse import spmatrix
import numpy as np
from pyomo.contrib.pynumero.sparse.base_block import BaseBlockVector, BaseBlockMatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
class DirectLinearSolverInterface(LinearSolverInterface, metaclass=ABCMeta):

    @abstractmethod
    def do_symbolic_factorization(self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool=True) -> LinearSolverResults:
        pass

    @abstractmethod
    def do_numeric_factorization(self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool=True) -> LinearSolverResults:
        pass

    @abstractmethod
    def do_back_solve(self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool=True) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        pass

    def solve(self, matrix: Union[spmatrix, BlockMatrix], rhs: Union[np.ndarray, BlockVector], raise_on_error: bool=True) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        symbolic_res = self.do_symbolic_factorization(matrix, raise_on_error=raise_on_error)
        if symbolic_res.status != LinearSolverStatus.successful:
            return (None, symbolic_res)
        numeric_res = self.do_numeric_factorization(matrix, raise_on_error=raise_on_error)
        if numeric_res.status != LinearSolverStatus.successful:
            return (None, numeric_res)
        return self.do_back_solve(rhs, raise_on_error=raise_on_error)