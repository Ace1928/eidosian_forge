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
def _initialize_grid_pool(self):
    """Sampling pool and sample grid."""
    self.sample_pool = []
    self.sample_grid = np.empty(np.append(self.grid_size, self.d), dtype=np.float32)
    self.sample_grid.fill(np.nan)