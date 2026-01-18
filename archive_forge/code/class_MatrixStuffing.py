import abc
from typing import List, Tuple
import numpy as np
from cvxpy.reductions.reduction import Reduction
class MatrixStuffing(Reduction):
    """Stuffs a problem into a standard form for a family of solvers."""
    __metaclass__ = abc.ABCMeta

    def apply(self, problem) -> None:
        """Returns a stuffed problem.

        The returned problem is a minimization problem in which every
        constraint in the problem has affine arguments that are expressed in
        the form A @ x + b.


        Parameters
        ----------
        problem: The problem to stuff; the arguments of every constraint
            must be affine

        Returns
        -------
        Problem
            The stuffed problem
        InverseData
            Data for solution retrieval
        """

    def invert(self, solution, inverse_data):
        raise NotImplementedError()

    def stuffed_objective(self, problem, inverse_data):
        raise NotImplementedError()