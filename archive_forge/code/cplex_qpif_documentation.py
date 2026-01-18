import numpy as np
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.cplex_conif import (
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
QP interface for the CPLEX solver