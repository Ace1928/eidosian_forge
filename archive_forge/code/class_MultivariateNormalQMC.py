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
class MultivariateNormalQMC:
    """QMC sampling from a multivariate Normal :math:`N(\\mu, \\Sigma)`.

    Parameters
    ----------
    mean : array_like (d,)
        The mean vector. Where ``d`` is the dimension.
    cov : array_like (d, d), optional
        The covariance matrix. If omitted, use `cov_root` instead.
        If both `cov` and `cov_root` are omitted, use the identity matrix.
    cov_root : array_like (d, d'), optional
        A root decomposition of the covariance matrix, where ``d'`` may be less
        than ``d`` if the covariance is not full rank. If omitted, use `cov`.
    inv_transform : bool, optional
        If True, use inverse transform instead of Box-Muller. Default is True.
    engine : QMCEngine, optional
        Quasi-Monte Carlo engine sampler. If None, `Sobol` is used.
    seed : {None, int, `numpy.random.Generator`}, optional
        Used only if `engine` is None.
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import qmc
    >>> dist = qmc.MultivariateNormalQMC(mean=[0, 5], cov=[[1, 0], [0, 1]])
    >>> sample = dist.random(512)
    >>> _ = plt.scatter(sample[:, 0], sample[:, 1])
    >>> plt.show()

    """

    def __init__(self, mean: npt.ArrayLike, cov: npt.ArrayLike | None=None, *, cov_root: npt.ArrayLike | None=None, inv_transform: bool=True, engine: QMCEngine | None=None, seed: SeedType=None) -> None:
        mean = np.array(mean, copy=False, ndmin=1)
        d = mean.shape[0]
        if cov is not None:
            cov = np.array(cov, copy=False, ndmin=2)
            if not mean.shape[0] == cov.shape[0]:
                raise ValueError('Dimension mismatch between mean and covariance.')
            if not np.allclose(cov, cov.transpose()):
                raise ValueError('Covariance matrix is not symmetric.')
            try:
                cov_root = np.linalg.cholesky(cov).transpose()
            except np.linalg.LinAlgError:
                eigval, eigvec = np.linalg.eigh(cov)
                if not np.all(eigval >= -1e-08):
                    raise ValueError('Covariance matrix not PSD.')
                eigval = np.clip(eigval, 0.0, None)
                cov_root = (eigvec * np.sqrt(eigval)).transpose()
        elif cov_root is not None:
            cov_root = np.atleast_2d(cov_root)
            if not mean.shape[0] == cov_root.shape[0]:
                raise ValueError('Dimension mismatch between mean and covariance.')
        else:
            cov_root = None
        self._inv_transform = inv_transform
        if not inv_transform:
            engine_dim = 2 * math.ceil(d / 2)
        else:
            engine_dim = d
        if engine is None:
            self.engine = Sobol(d=engine_dim, scramble=True, bits=30, seed=seed)
        elif isinstance(engine, QMCEngine):
            if engine.d != engine_dim:
                raise ValueError('Dimension of `engine` must be consistent with dimensions of mean and covariance. If `inv_transform` is False, it must be an even number.')
            self.engine = engine
        else:
            raise ValueError('`engine` must be an instance of `scipy.stats.qmc.QMCEngine` or `None`.')
        self._mean = mean
        self._corr_matrix = cov_root
        self._d = d

    def random(self, n: IntNumber=1) -> np.ndarray:
        """Draw `n` QMC samples from the multivariate Normal.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample.

        """
        base_samples = self._standard_normal_samples(n)
        return self._correlate(base_samples)

    def _correlate(self, base_samples: np.ndarray) -> np.ndarray:
        if self._corr_matrix is not None:
            return base_samples @ self._corr_matrix + self._mean
        else:
            return base_samples + self._mean

    def _standard_normal_samples(self, n: IntNumber=1) -> np.ndarray:
        """Draw `n` QMC samples from the standard Normal :math:`N(0, I_d)`.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sample.

        """
        samples = self.engine.random(n)
        if self._inv_transform:
            return stats.norm.ppf(0.5 + (1 - 1e-10) * (samples - 0.5))
        else:
            even = np.arange(0, samples.shape[-1], 2)
            Rs = np.sqrt(-2 * np.log(samples[:, even]))
            thetas = 2 * math.pi * samples[:, 1 + even]
            cos = np.cos(thetas)
            sin = np.sin(thetas)
            transf_samples = np.stack([Rs * cos, Rs * sin], -1).reshape(n, -1)
            return transf_samples[:, :self._d]