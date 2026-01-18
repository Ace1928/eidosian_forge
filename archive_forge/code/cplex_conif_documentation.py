from collections import namedtuple
from operator import attrgetter
import numpy as np
from scipy.sparse import dok_matrix
import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
Adds SOC constraint to the model using the data from mat and vec.

        Parameters
        ----------
        model : CPLEX model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        mat : SciPy COO matrix
            The matrix representing the constraints.
        vec : NDArray
            The RHS part of the constraints.

        Returns
        -------
        tuple
            A tuple of (a new quadratic constraint index, a list of new
            supporting linear constr indices, and a list of new
            supporting variable indices).
        