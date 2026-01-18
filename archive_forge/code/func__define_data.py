import logging
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, Union
import numpy as np
from scipy.sparse import dok_matrix
import cvxpy.settings as s
from cvxpy import Zero
from cvxpy.constraints import SOC, ExpCone, NonNeg
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
def _define_data(self, data: Dict[str, Any]) -> Tuple:
    """Define data parts from the data reference."""
    c = data[s.C]
    b = data[s.B]
    A = dok_matrix(data[s.A])
    data[s.A] = A
    dims = dims_to_solver_dict(data[s.DIMS])
    return (A, b, c, dims)