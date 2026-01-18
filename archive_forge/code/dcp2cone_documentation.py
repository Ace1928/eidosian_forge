from typing import Tuple
from cvxpy import problems
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS as cone_canon_methods
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.qp2quad_form.canonicalizers import QUAD_CANON_METHODS as quad_canon_methods
Canonicalize an expression, w.r.t. canonicalized arguments.

        Parameters
        ----------
        expr : The expression tree to canonicalize.
        args : The canonicalized arguments of expr.
        affine_above : The path up to the root node is all affine atoms.

        Returns
        -------
        A tuple of the canonicalized expression and generated constraints.
        