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