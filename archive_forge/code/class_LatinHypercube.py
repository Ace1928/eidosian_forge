from __future__ import annotations
import copy
import math
import numbers
import os
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
import numpy as np
import scipy.stats as stats
from scipy._lib._util import rng_integers, _rng_spawn
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance, Voronoi
from scipy.special import gammainc
from ._sobol import (
from ._qmc_cy import (
class LatinHypercube(QMCEngine):
    """Latin hypercube sampling (LHS).

    A Latin hypercube sample [1]_ generates :math:`n` points in
    :math:`[0,1)^{d}`. Each univariate marginal distribution is stratified,
    placing exactly one point in :math:`[j/n, (j+1)/n)` for
    :math:`j=0,1,...,n-1`. They are still applicable when :math:`n << d`.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    scramble : bool, optional
        When False, center samples within cells of a multi-dimensional grid.
        Otherwise, samples are randomly placed within cells of the grid.

        .. note::
            Setting ``scramble=False`` does not ensure deterministic output.
            For that, use the `seed` parameter.

        Default is True.

        .. versionadded:: 1.10.0

    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.8.0
        .. versionchanged:: 1.10.0
            Add ``lloyd``.

    strength : {1, 2}, optional
        Strength of the LHS. ``strength=1`` produces a plain LHS while
        ``strength=2`` produces an orthogonal array based LHS of strength 2
        [7]_, [8]_. In that case, only ``n=p**2`` points can be sampled,
        with ``p`` a prime number. It also constrains ``d <= p + 1``.
        Default is 1.

        .. versionadded:: 1.8.0

    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----

    When LHS is used for integrating a function :math:`f` over :math:`n`,
    LHS is extremely effective on integrands that are nearly additive [2]_.
    With a LHS of :math:`n` points, the variance of the integral is always
    lower than plain MC on :math:`n-1` points [3]_. There is a central limit
    theorem for LHS on the mean and variance of the integral [4]_, but not
    necessarily for optimized LHS due to the randomization.

    :math:`A` is called an orthogonal array of strength :math:`t` if in each
    n-row-by-t-column submatrix of :math:`A`: all :math:`p^t` possible
    distinct rows occur the same number of times. The elements of :math:`A`
    are in the set :math:`\\{0, 1, ..., p-1\\}`, also called symbols.
    The constraint that :math:`p` must be a prime number is to allow modular
    arithmetic. Increasing strength adds some symmetry to the sub-projections
    of a sample. With strength 2, samples are symmetric along the diagonals of
    2D sub-projections. This may be undesirable, but on the other hand, the
    sample dispersion is improved.

    Strength 1 (plain LHS) brings an advantage over strength 0 (MC) and
    strength 2 is a useful increment over strength 1. Going to strength 3 is
    a smaller increment and scrambled QMC like Sobol', Halton are more
    performant [7]_.

    To create a LHS of strength 2, the orthogonal array :math:`A` is
    randomized by applying a random, bijective map of the set of symbols onto
    itself. For example, in column 0, all 0s might become 2; in column 1,
    all 0s might become 1, etc.
    Then, for each column :math:`i` and symbol :math:`j`, we add a plain,
    one-dimensional LHS of size :math:`p` to the subarray where
    :math:`A^i = j`. The resulting matrix is finally divided by :math:`p`.

    References
    ----------
    .. [1] Mckay et al., "A Comparison of Three Methods for Selecting Values
       of Input Variables in the Analysis of Output from a Computer Code."
       Technometrics, 1979.
    .. [2] M. Stein, "Large sample properties of simulations using Latin
       hypercube sampling." Technometrics 29, no. 2: 143-151, 1987.
    .. [3] A. B. Owen, "Monte Carlo variance of scrambled net quadrature."
       SIAM Journal on Numerical Analysis 34, no. 5: 1884-1910, 1997
    .. [4]  Loh, W.-L. "On Latin hypercube sampling." The annals of statistics
       24, no. 5: 2058-2080, 1996.
    .. [5] Fang et al. "Design and modeling for computer experiments".
       Computer Science and Data Analysis Series, 2006.
    .. [6] Damblin et al., "Numerical studies of space filling designs:
       optimization of Latin Hypercube Samples and subprojection properties."
       Journal of Simulation, 2013.
    .. [7] A. B. Owen , "Orthogonal arrays for computer experiments,
       integration and visualization." Statistica Sinica, 1992.
    .. [8] B. Tang, "Orthogonal Array-Based Latin Hypercubes."
       Journal of the American Statistical Association, 1993.
    .. [9] Susan K. Seaholm et al. "Latin hypercube sampling and the
       sensitivity analysis of a Monte Carlo epidemic model".
       Int J Biomed Comput, 23(1-2), 97-112,
       :doi:`10.1016/0020-7101(88)90067-0`, 1988.

    Examples
    --------
    In [9]_, a Latin Hypercube sampling strategy was used to sample a
    parameter space to study the importance of each parameter of an epidemic
    model. Such analysis is also called a sensitivity analysis.

    Since the dimensionality of the problem is high (6), it is computationally
    expensive to cover the space. When numerical experiments are costly,
    QMC enables analysis that may not be possible if using a grid.

    The six parameters of the model represented the probability of illness,
    the probability of withdrawal, and four contact probabilities,
    The authors assumed uniform distributions for all parameters and generated
    50 samples.

    Using `scipy.stats.qmc.LatinHypercube` to replicate the protocol, the
    first step is to create a sample in the unit hypercube:

    >>> from scipy.stats import qmc
    >>> sampler = qmc.LatinHypercube(d=6)
    >>> sample = sampler.random(n=50)

    Then the sample can be scaled to the appropriate bounds:

    >>> l_bounds = [0.000125, 0.01, 0.0025, 0.05, 0.47, 0.7]
    >>> u_bounds = [0.000375, 0.03, 0.0075, 0.15, 0.87, 0.9]
    >>> sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    Such a sample was used to run the model 50 times, and a polynomial
    response surface was constructed. This allowed the authors to study the
    relative importance of each parameter across the range of
    possibilities of every other parameter.
    In this computer experiment, they showed a 14-fold reduction in the number
    of samples required to maintain an error below 2% on their response surface
    when compared to a grid sampling.

    Below are other examples showing alternative ways to construct LHS
    with even better coverage of the space.

    Using a base LHS as a baseline.

    >>> sampler = qmc.LatinHypercube(d=2)
    >>> sample = sampler.random(n=5)
    >>> qmc.discrepancy(sample)
    0.0196...  # random

    Use the `optimization` keyword argument to produce a LHS with
    lower discrepancy at higher computational cost.

    >>> sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
    >>> sample = sampler.random(n=5)
    >>> qmc.discrepancy(sample)
    0.0176...  # random

    Use the `strength` keyword argument to produce an orthogonal array based
    LHS of strength 2. In this case, the number of sample points must be the
    square of a prime number.

    >>> sampler = qmc.LatinHypercube(d=2, strength=2)
    >>> sample = sampler.random(n=9)
    >>> qmc.discrepancy(sample)
    0.00526...  # random

    Options could be combined to produce an optimized centered
    orthogonal array based LHS. After optimization, the result would not
    be guaranteed to be of strength 2.

    """

    def __init__(self, d: IntNumber, *, scramble: bool=True, strength: int=1, optimization: Literal['random-cd', 'lloyd'] | None=None, seed: SeedType=None) -> None:
        self._init_quad = {'d': d, 'scramble': True, 'strength': strength, 'optimization': optimization}
        super().__init__(d=d, seed=seed, optimization=optimization)
        self.scramble = scramble
        lhs_method_strength = {1: self._random_lhs, 2: self._random_oa_lhs}
        try:
            self.lhs_method: Callable = lhs_method_strength[strength]
        except KeyError as exc:
            message = f'{strength!r} is not a valid strength. It must be one of {set(lhs_method_strength)!r}'
            raise ValueError(message) from exc

    def _random(self, n: IntNumber=1, *, workers: IntNumber=1) -> np.ndarray:
        lhs = self.lhs_method(n)
        return lhs

    def _random_lhs(self, n: IntNumber=1) -> np.ndarray:
        """Base LHS algorithm."""
        if not self.scramble:
            samples: np.ndarray | float = 0.5
        else:
            samples = self.rng.uniform(size=(n, self.d))
        perms = np.tile(np.arange(1, n + 1), (self.d, 1))
        for i in range(self.d):
            self.rng.shuffle(perms[i, :])
        perms = perms.T
        samples = (perms - samples) / n
        return samples

    def _random_oa_lhs(self, n: IntNumber=4) -> np.ndarray:
        """Orthogonal array based LHS of strength 2."""
        p = np.sqrt(n).astype(int)
        n_row = p ** 2
        n_col = p + 1
        primes = primes_from_2_to(p + 1)
        if p not in primes or n != n_row:
            raise ValueError(f'n is not the square of a prime number. Close values are {primes[-2:] ** 2}')
        if self.d > p + 1:
            raise ValueError('n is too small for d. Must be n > (d-1)**2')
        oa_sample = np.zeros(shape=(n_row, n_col), dtype=int)
        arrays = np.tile(np.arange(p), (2, 1))
        oa_sample[:, :2] = np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, 2)
        for p_ in range(1, p):
            oa_sample[:, 2 + p_ - 1] = np.mod(oa_sample[:, 0] + p_ * oa_sample[:, 1], p)
        oa_sample_ = np.empty(shape=(n_row, n_col), dtype=int)
        for j in range(n_col):
            perms = self.rng.permutation(p)
            oa_sample_[:, j] = perms[oa_sample[:, j]]
        oa_lhs_sample = np.zeros(shape=(n_row, n_col))
        lhs_engine = LatinHypercube(d=1, scramble=self.scramble, strength=1, seed=self.rng)
        for j in range(n_col):
            for k in range(p):
                idx = oa_sample[:, j] == k
                lhs = lhs_engine.random(p).flatten()
                oa_lhs_sample[:, j][idx] = lhs + oa_sample[:, j][idx]
                lhs_engine = lhs_engine.reset()
        oa_lhs_sample /= p
        return oa_lhs_sample[:, :self.d]