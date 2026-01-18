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
def _hypersphere_volume_sample(self, center: np.ndarray, radius: DecimalNumber, candidates: IntNumber=1) -> np.ndarray:
    """Uniform sampling within hypersphere."""
    x = self.rng.standard_normal(size=(candidates, self.d))
    ssq = np.sum(x ** 2, axis=1)
    fr = radius * gammainc(self.d / 2, ssq / 2) ** (1 / self.d) / np.sqrt(ssq)
    fr_tiled = np.tile(fr.reshape(-1, 1), (1, self.d))
    p = center + np.multiply(x, fr_tiled)
    return p