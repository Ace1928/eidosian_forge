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
class PoissonDisk(QMCEngine):
    """Poisson disk sampling.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    radius : float
        Minimal distance to keep between points when sampling new candidates.
    hypersphere : {"volume", "surface"}, optional
        Sampling strategy to generate potential candidates to be added in the
        final sample. Default is "volume".

        * ``volume``: original Bridson algorithm as described in [1]_.
          New candidates are sampled *within* the hypersphere.
        * ``surface``: only sample the surface of the hypersphere.
    ncandidates : int
        Number of candidates to sample per iteration. More candidates result
        in a denser sampling as more candidates can be accepted per iteration.
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

        .. versionadded:: 1.10.0
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----
    Poisson disk sampling is an iterative sampling strategy. Starting from
    a seed sample, `ncandidates` are sampled in the hypersphere
    surrounding the seed. Candidates below a certain `radius` or outside the
    domain are rejected. New samples are added in a pool of sample seed. The
    process stops when the pool is empty or when the number of required
    samples is reached.

    The maximum number of point that a sample can contain is directly linked
    to the `radius`. As the dimension of the space increases, a higher radius
    spreads the points further and help overcome the curse of dimensionality.
    See the :ref:`quasi monte carlo tutorial <quasi-monte-carlo>` for more
    details.

    .. warning::

       The algorithm is more suitable for low dimensions and sampling size
       due to its iterative nature and memory requirements.
       Selecting a small radius with a high dimension would
       mean that the space could contain more samples than using lower
       dimension or a bigger radius.

    Some code taken from [2]_, written consent given on 31.03.2021
    by the original author, Shamis, for free use in SciPy under
    the 3-clause BSD.

    References
    ----------
    .. [1] Robert Bridson, "Fast Poisson Disk Sampling in Arbitrary
       Dimensions." SIGGRAPH, 2007.
    .. [2] `StackOverflow <https://stackoverflow.com/questions/66047540>`__.

    Examples
    --------
    Generate a 2D sample using a `radius` of 0.2.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.collections import PatchCollection
    >>> from scipy.stats import qmc
    >>>
    >>> rng = np.random.default_rng()
    >>> radius = 0.2
    >>> engine = qmc.PoissonDisk(d=2, radius=radius, seed=rng)
    >>> sample = engine.random(20)

    Visualizing the 2D sample and showing that no points are closer than
    `radius`. ``radius/2`` is used to visualize non-intersecting circles.
    If two samples are exactly at `radius` from each other, then their circle
    of radius ``radius/2`` will touch.

    >>> fig, ax = plt.subplots()
    >>> _ = ax.scatter(sample[:, 0], sample[:, 1])
    >>> circles = [plt.Circle((xi, yi), radius=radius/2, fill=False)
    ...            for xi, yi in sample]
    >>> collection = PatchCollection(circles, match_original=True)
    >>> ax.add_collection(collection)
    >>> _ = ax.set(aspect='equal', xlabel=r'$x_1$', ylabel=r'$x_2$',
    ...            xlim=[0, 1], ylim=[0, 1])
    >>> plt.show()

    Such visualization can be seen as circle packing: how many circle can
    we put in the space. It is a np-hard problem. The method `fill_space`
    can be used to add samples until no more samples can be added. This is
    a hard problem and parameters may need to be adjusted manually. Beware of
    the dimension: as the dimensionality increases, the number of samples
    required to fill the space increases exponentially
    (curse-of-dimensionality).

    """

    def __init__(self, d: IntNumber, *, radius: DecimalNumber=0.05, hypersphere: Literal['volume', 'surface']='volume', ncandidates: IntNumber=30, optimization: Literal['random-cd', 'lloyd'] | None=None, seed: SeedType=None) -> None:
        self._init_quad = {'d': d, 'radius': radius, 'hypersphere': hypersphere, 'ncandidates': ncandidates, 'optimization': optimization}
        super().__init__(d=d, optimization=optimization, seed=seed)
        hypersphere_sample = {'volume': self._hypersphere_volume_sample, 'surface': self._hypersphere_surface_sample}
        try:
            self.hypersphere_method = hypersphere_sample[hypersphere]
        except KeyError as exc:
            message = f'{hypersphere!r} is not a valid hypersphere sampling method. It must be one of {set(hypersphere_sample)!r}'
            raise ValueError(message) from exc
        self.radius_factor = 2 if hypersphere == 'volume' else 1.001
        self.radius = radius
        self.radius_squared = self.radius ** 2
        self.ncandidates = ncandidates
        with np.errstate(divide='ignore'):
            self.cell_size = self.radius / np.sqrt(self.d)
            self.grid_size = np.ceil(np.ones(self.d) / self.cell_size).astype(int)
        self._initialize_grid_pool()

    def _initialize_grid_pool(self):
        """Sampling pool and sample grid."""
        self.sample_pool = []
        self.sample_grid = np.empty(np.append(self.grid_size, self.d), dtype=np.float32)
        self.sample_grid.fill(np.nan)

    def _random(self, n: IntNumber=1, *, workers: IntNumber=1) -> np.ndarray:
        """Draw `n` in the interval ``[0, 1]``.

        Note that it can return fewer samples if the space is full.
        See the note section of the class.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        if n == 0 or self.d == 0:
            return np.empty((n, self.d))

        def in_limits(sample: np.ndarray) -> bool:
            return sample.max() <= 1.0 and sample.min() >= 0.0

        def in_neighborhood(candidate: np.ndarray, n: int=2) -> bool:
            """
            Check if there are samples closer than ``radius_squared`` to the
            `candidate` sample.
            """
            indices = (candidate / self.cell_size).astype(int)
            ind_min = np.maximum(indices - n, np.zeros(self.d, dtype=int))
            ind_max = np.minimum(indices + n + 1, self.grid_size)
            if not np.isnan(self.sample_grid[tuple(indices)][0]):
                return True
            a = [slice(ind_min[i], ind_max[i]) for i in range(self.d)]
            with np.errstate(invalid='ignore'):
                if np.any(np.sum(np.square(candidate - self.sample_grid[tuple(a)]), axis=self.d) < self.radius_squared):
                    return True
            return False

        def add_sample(candidate: np.ndarray) -> None:
            self.sample_pool.append(candidate)
            indices = (candidate / self.cell_size).astype(int)
            self.sample_grid[tuple(indices)] = candidate
            curr_sample.append(candidate)
        curr_sample: list[np.ndarray] = []
        if len(self.sample_pool) == 0:
            add_sample(self.rng.random(self.d))
            num_drawn = 1
        else:
            num_drawn = 0
        while len(self.sample_pool) and num_drawn < n:
            idx_center = rng_integers(self.rng, len(self.sample_pool))
            center = self.sample_pool[idx_center]
            del self.sample_pool[idx_center]
            candidates = self.hypersphere_method(center, self.radius * self.radius_factor, self.ncandidates)
            for candidate in candidates:
                if in_limits(candidate) and (not in_neighborhood(candidate)):
                    add_sample(candidate)
                    num_drawn += 1
                    if num_drawn >= n:
                        break
        self.num_generated += num_drawn
        return np.array(curr_sample)

    def fill_space(self) -> np.ndarray:
        """Draw ``n`` samples in the interval ``[0, 1]``.

        Unlike `random`, this method will try to add points until
        the space is full. Depending on ``candidates`` (and to a lesser extent
        other parameters), some empty areas can still be present in the sample.

        .. warning::

           This can be extremely slow in high dimensions or if the
           ``radius`` is very small-with respect to the dimensionality.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        return self.random(np.inf)

    def reset(self) -> PoissonDisk:
        """Reset the engine to base state.

        Returns
        -------
        engine : PoissonDisk
            Engine reset to its base state.

        """
        super().reset()
        self._initialize_grid_pool()
        return self

    def _hypersphere_volume_sample(self, center: np.ndarray, radius: DecimalNumber, candidates: IntNumber=1) -> np.ndarray:
        """Uniform sampling within hypersphere."""
        x = self.rng.standard_normal(size=(candidates, self.d))
        ssq = np.sum(x ** 2, axis=1)
        fr = radius * gammainc(self.d / 2, ssq / 2) ** (1 / self.d) / np.sqrt(ssq)
        fr_tiled = np.tile(fr.reshape(-1, 1), (1, self.d))
        p = center + np.multiply(x, fr_tiled)
        return p

    def _hypersphere_surface_sample(self, center: np.ndarray, radius: DecimalNumber, candidates: IntNumber=1) -> np.ndarray:
        """Uniform sampling on the hypersphere's surface."""
        vec = self.rng.standard_normal(size=(candidates, self.d))
        vec /= np.linalg.norm(vec, axis=1)[:, None]
        p = center + np.multiply(vec, radius)
        return p