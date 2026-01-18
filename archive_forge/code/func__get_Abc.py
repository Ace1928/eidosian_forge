import numpy as np
import scipy.sparse as sps
from warnings import warn
from ._optimize import OptimizeWarning
from scipy.optimize._remove_redundancy import (
from collections import namedtuple
def _get_Abc(lp, c0):
    """
    Given a linear programming problem of the form:

    Minimize::

        c @ x

    Subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub

    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.

    Return the problem in standard form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    by adding slack variables and making variable substitutions as necessary.

    Parameters
    ----------
    lp : A `scipy.optimize._linprog_util._LPProblem` consisting of the following fields:

        c : 1D array
            The coefficients of the linear objective function to be minimized.
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        bounds : 2D array
            The bounds of ``x``, lower bounds in the 1st column, upper
            bounds in the 2nd column. The bounds are possibly tightened
            by the presolve procedure.
        x0 : 1D array, optional
            Guess values of the decision variables, which will be refined by
            the optimization algorithm. This argument is currently used only by the
            'revised simplex' method, and can only be used if `x0` represents a
            basic feasible solution.

    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables.

    Returns
    -------
    A : 2-D array
        2-D array such that ``A`` @ ``x``, gives the values of the equality
        constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in A (for standard form problem).
    c : 1-D array
        Coefficients of the linear objective function to be minimized (for
        standard form problem).
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables.
    x0 : 1-D array
        Starting values of the independent variables, which will be refined by
        the optimization algorithm

    References
    ----------
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.

    """
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality = lp
    if sps.issparse(A_eq):
        sparse = True
        A_eq = sps.csr_matrix(A_eq)
        A_ub = sps.csr_matrix(A_ub)

        def hstack(blocks):
            return sps.hstack(blocks, format='csr')

        def vstack(blocks):
            return sps.vstack(blocks, format='csr')
        zeros = sps.csr_matrix
        eye = sps.eye
    else:
        sparse = False
        hstack = np.hstack
        vstack = np.vstack
        zeros = np.zeros
        eye = np.eye
    bounds = np.array(bounds, copy=True)
    lbs = bounds[:, 0]
    ubs = bounds[:, 1]
    m_ub, n_ub = A_ub.shape
    lb_none = np.equal(lbs, -np.inf)
    ub_none = np.equal(ubs, np.inf)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)
    l_nolb_someub = np.logical_and(lb_none, ub_some)
    i_nolb = np.nonzero(l_nolb_someub)[0]
    lbs[l_nolb_someub], ubs[l_nolb_someub] = (-ubs[l_nolb_someub], -lbs[l_nolb_someub])
    lb_none = np.equal(lbs, -np.inf)
    ub_none = np.equal(ubs, np.inf)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)
    c[i_nolb] *= -1
    if x0 is not None:
        x0[i_nolb] *= -1
    if len(i_nolb) > 0:
        if A_ub.shape[0] > 0:
            A_ub[:, i_nolb] *= -1
        if A_eq.shape[0] > 0:
            A_eq[:, i_nolb] *= -1
    i_newub, = ub_some.nonzero()
    ub_newub = ubs[ub_some]
    n_bounds = len(i_newub)
    if n_bounds > 0:
        shape = (n_bounds, A_ub.shape[1])
        if sparse:
            idxs = (np.arange(n_bounds), i_newub)
            A_ub = vstack((A_ub, sps.csr_matrix((np.ones(n_bounds), idxs), shape=shape)))
        else:
            A_ub = vstack((A_ub, np.zeros(shape)))
            A_ub[np.arange(m_ub, A_ub.shape[0]), i_newub] = 1
        b_ub = np.concatenate((b_ub, np.zeros(n_bounds)))
        b_ub[m_ub:] = ub_newub
    A1 = vstack((A_ub, A_eq))
    b = np.concatenate((b_ub, b_eq))
    c = np.concatenate((c, np.zeros((A_ub.shape[0],))))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros((A_ub.shape[0],))))
    l_free = np.logical_and(lb_none, ub_none)
    i_free = np.nonzero(l_free)[0]
    n_free = len(i_free)
    c = np.concatenate((c, np.zeros(n_free)))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros(n_free)))
    A1 = hstack((A1[:, :n_ub], -A1[:, i_free]))
    c[n_ub:n_ub + n_free] = -c[i_free]
    if x0 is not None:
        i_free_neg = x0[i_free] < 0
        x0[np.arange(n_ub, A1.shape[1])[i_free_neg]] = -x0[i_free[i_free_neg]]
        x0[i_free[i_free_neg]] = 0
    A2 = vstack([eye(A_ub.shape[0]), zeros((A_eq.shape[0], A_ub.shape[0]))])
    A = hstack([A1, A2])
    i_shift = np.nonzero(lb_some)[0]
    lb_shift = lbs[lb_some].astype(float)
    c0 += np.sum(lb_shift * c[i_shift])
    if sparse:
        b = b.reshape(-1, 1)
        A = A.tocsc()
        b -= (A[:, i_shift] * sps.diags(lb_shift)).sum(axis=1)
        b = b.ravel()
    else:
        b -= (A[:, i_shift] * lb_shift).sum(axis=1)
    if x0 is not None:
        x0[i_shift] -= lb_shift
    return (A, b, c, c0, x0)