import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse
class PreparedConstraint:
    """Constraint prepared from a user defined constraint.

    On creation it will check whether a constraint definition is valid and
    the initial point is feasible. If created successfully, it will contain
    the attributes listed below.

    Parameters
    ----------
    constraint : {NonlinearConstraint, LinearConstraint`, Bounds}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables.
    sparse_jacobian : bool or None, optional
        If bool, then the Jacobian of the constraint will be converted
        to the corresponded format if necessary. If None (default), such
        conversion is not made.
    finite_diff_bounds : 2-tuple, optional
        Lower and upper bounds on the independent variables for the finite
        difference approximation, if applicable. Defaults to no bounds.

    Attributes
    ----------
    fun : {VectorFunction, LinearVectorFunction, IdentityVectorFunction}
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.
    keep_feasible : ndarray
         Array indicating which components must be kept feasible with a size
         equal to the number of the constraints.
    """

    def __init__(self, constraint, x0, sparse_jacobian=None, finite_diff_bounds=(-np.inf, np.inf)):
        if isinstance(constraint, NonlinearConstraint):
            fun = VectorFunction(constraint.fun, x0, constraint.jac, constraint.hess, constraint.finite_diff_rel_step, constraint.finite_diff_jac_sparsity, finite_diff_bounds, sparse_jacobian)
        elif isinstance(constraint, LinearConstraint):
            fun = LinearVectorFunction(constraint.A, x0, sparse_jacobian)
        elif isinstance(constraint, Bounds):
            fun = IdentityVectorFunction(x0, sparse_jacobian)
        else:
            raise ValueError('`constraint` of an unknown type is passed.')
        m = fun.m
        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)
        keep_feasible = np.asarray(constraint.keep_feasible, dtype=bool)
        lb = np.broadcast_to(lb, m)
        ub = np.broadcast_to(ub, m)
        keep_feasible = np.broadcast_to(keep_feasible, m)
        if keep_feasible.shape != (m,):
            raise ValueError('`keep_feasible` has a wrong shape.')
        mask = keep_feasible & (lb != ub)
        f0 = fun.f
        if np.any(f0[mask] < lb[mask]) or np.any(f0[mask] > ub[mask]):
            raise ValueError('`x0` is infeasible with respect to some inequality constraint with `keep_feasible` set to True.')
        self.fun = fun
        self.bounds = (lb, ub)
        self.keep_feasible = keep_feasible

    def violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by, for each of the
            constraints specified by `PreparedConstraint.fun`.
        """
        with catch_warnings():
            filterwarnings('ignore', 'delta_grad', UserWarning)
            ev = self.fun.fun(np.asarray(x))
        excess_lb = np.maximum(self.bounds[0] - ev, 0)
        excess_ub = np.maximum(ev - self.bounds[1], 0)
        return excess_lb + excess_ub