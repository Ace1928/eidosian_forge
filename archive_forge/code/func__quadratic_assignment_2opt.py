import numpy as np
import operator
from . import (linear_sum_assignment, OptimizeResult)
from ._optimize import _check_unknown_options
from scipy._lib._util import check_random_state
import itertools
def _quadratic_assignment_2opt(A, B, maximize=False, rng=None, partial_match=None, partial_guess=None, **unknown_options):
    """Solve the quadratic assignment problem (approximately).

    This function solves the Quadratic Assignment Problem (QAP) and the
    Graph Matching Problem (GMP) using the 2-opt algorithm [1]_.

    Quadratic assignment solves problems of the following form:

    .. math::

        \\min_P & \\ {\\ \\text{trace}(A^T P B P^T)}\\\\
        \\mbox{s.t. } & {P \\ \\epsilon \\ \\mathcal{P}}\\\\

    where :math:`\\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices.

    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    edge weight differences.

    Note that the quadratic assignment problem is NP-hard. The results given
    here are approximations and are not guaranteed to be optimal.

    Parameters
    ----------
    A : 2-D array, square
        The square matrix :math:`A` in the objective function above.
    B : 2-D array, square
        The square matrix :math:`B` in the objective function above.
    method :  str in {'faq', '2opt'} (default: 'faq')
        The algorithm used to solve the problem. This is the method-specific
        documentation for '2opt'.
        :ref:`'faq' <optimize.qap-faq>` is also available.

    Options
    -------
    maximize : bool (default: False)
        Maximizes the objective function if ``True``.
    rng : {None, int, `numpy.random.Generator`,
           `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    partial_match : 2-D array of integers, optional (default: None)
        Fixes part of the matching. Also known as a "seed" [2]_.

        Each row of `partial_match` specifies a pair of matched nodes: node
        ``partial_match[i, 0]`` of `A` is matched to node
        ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``,
        where ``m`` is not greater than the number of nodes, :math:`n`.
    partial_guess : 2-D array of integers, optional (default: None)
        A guess for the matching between the two matrices. Unlike
        `partial_match`, `partial_guess` does not fix the indices; they are
        still free to be optimized.

        Each row of `partial_guess` specifies a pair of matched nodes: node
        ``partial_guess[i, 0]`` of `A` is matched to node
        ``partial_guess[i, 1]`` of `B`. The array has shape ``(m, 2)``,
        where ``m`` is not greater than the number of nodes, :math:`n`.

    Returns
    -------
    res : OptimizeResult
        `OptimizeResult` containing the following fields.

        col_ind : 1-D array
            Column indices corresponding to the best permutation found of the
            nodes of `B`.
        fun : float
            The objective value of the solution.
        nit : int
            The number of iterations performed during optimization.

    Notes
    -----
    This is a greedy algorithm that works similarly to bubble sort: beginning
    with an initial permutation, it iteratively swaps pairs of indices to
    improve the objective function until no such improvements are possible.

    References
    ----------
    .. [1] "2-opt," Wikipedia.
           https://en.wikipedia.org/wiki/2-opt

    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,
           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):
           203-215, https://doi.org/10.1016/j.patcog.2018.09.014

    """
    _check_unknown_options(unknown_options)
    rng = check_random_state(rng)
    A, B, partial_match = _common_input_validation(A, B, partial_match)
    N = len(A)
    if N == 0 or partial_match.shape[0] == N:
        score = _calc_score(A, B, partial_match[:, 1])
        res = {'col_ind': partial_match[:, 1], 'fun': score, 'nit': 0}
        return OptimizeResult(res)
    if partial_guess is None:
        partial_guess = np.array([[], []]).T
    partial_guess = np.atleast_2d(partial_guess).astype(int)
    msg = None
    if partial_guess.shape[0] > A.shape[0]:
        msg = '`partial_guess` can have only as many entries as there are nodes'
    elif partial_guess.shape[1] != 2:
        msg = '`partial_guess` must have two columns'
    elif partial_guess.ndim != 2:
        msg = '`partial_guess` must have exactly two dimensions'
    elif (partial_guess < 0).any():
        msg = '`partial_guess` must contain only positive indices'
    elif (partial_guess >= len(A)).any():
        msg = '`partial_guess` entries must be less than number of nodes'
    elif not len(set(partial_guess[:, 0])) == len(partial_guess[:, 0]) or not len(set(partial_guess[:, 1])) == len(partial_guess[:, 1]):
        msg = '`partial_guess` column entries must be unique'
    if msg is not None:
        raise ValueError(msg)
    fixed_rows = None
    if partial_match.size or partial_guess.size:
        guess_rows = np.zeros(N, dtype=bool)
        guess_cols = np.zeros(N, dtype=bool)
        fixed_rows = np.zeros(N, dtype=bool)
        fixed_cols = np.zeros(N, dtype=bool)
        perm = np.zeros(N, dtype=int)
        rg, cg = partial_guess.T
        guess_rows[rg] = True
        guess_cols[cg] = True
        perm[guess_rows] = cg
        rf, cf = partial_match.T
        fixed_rows[rf] = True
        fixed_cols[cf] = True
        perm[fixed_rows] = cf
        random_rows = ~fixed_rows & ~guess_rows
        random_cols = ~fixed_cols & ~guess_cols
        perm[random_rows] = rng.permutation(np.arange(N)[random_cols])
    else:
        perm = rng.permutation(np.arange(N))
    best_score = _calc_score(A, B, perm)
    i_free = np.arange(N)
    if fixed_rows is not None:
        i_free = i_free[~fixed_rows]
    better = operator.gt if maximize else operator.lt
    n_iter = 0
    done = False
    while not done:
        for i, j in itertools.combinations_with_replacement(i_free, 2):
            n_iter += 1
            perm[i], perm[j] = (perm[j], perm[i])
            score = _calc_score(A, B, perm)
            if better(score, best_score):
                best_score = score
                break
            perm[i], perm[j] = (perm[j], perm[i])
        else:
            done = True
    res = {'col_ind': perm, 'fun': best_score, 'nit': n_iter}
    return OptimizeResult(res)