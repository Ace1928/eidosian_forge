import numpy as np
import scipy as sp
from cvxpy import settings as s
from cvxpy.constraints.exponential import ExpCone as ExpCone_obj
from cvxpy.constraints.nonpos import NonNeg as NonNeg_obj
from cvxpy.constraints.power import PowCone3D as PowCone_obj
from cvxpy.constraints.psd import PSD as PSD_obj
from cvxpy.constraints.second_order import SOC as SOC_obj
from cvxpy.constraints.zero import Zero as Zero_obj
from cvxpy.reductions.solution import Solution

        "prob" is a ParamConeProg which represents

            (Aff)   min{ c.T @ x : A @ x + b in K,
                                  x[bools] in {0, 1}, x[ints] in Z } + d.

        We return data for an equivalent problem

            (Dir)   min{ f @ y : G @ y <=_{K_aff} h, y in K_dir
                                 y[bools] in {0, 1}, y[ints] in Z } + d,

        where

            (1) K_aff is built from cone types specified in "affine" (a list of strings),
            (2) a primal solution for (Dir) can be mapped back to a primal solution
                for (Aff) by selecting the leading ``c.size`` block of y's components.

        In the returned dict "data", data[s.A] = G, data[s.B] = h, data[s.C] = f,
        data['K_aff'] = K_aff, data['K_dir'] = K_dir, data[s.BOOL_IDX] = bools,
        and data[s.INT_IDX] = ints. The rows of G are ordered according to ZERO, then
        (as applicable) NONNEG, SOC, and EXP. If  "c" is the objective vector in (Aff),
        then ``y[:c.size]`` should contain the optimal solution to (Aff). The columns of
        G correspond first to variables in cones FREE, then NONNEG, then SOC, then EXP.
        The length of the free cone is equal to ``c.size``.

        Assumptions
        -----------
        The function call ``c, d, A, b = prob.apply_parameters()`` returns (A,b) with
        rows formatted first for the zero cone, then for the nonnegative orthant, then
        second order cones, then the exponential cone. Removing this assumption will
        require adding additional data to ParamConeProg objects.
        