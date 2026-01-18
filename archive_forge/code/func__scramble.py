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
def _scramble(self) -> None:
    """Scramble the sequence using LMS+shift."""
    self._shift = np.dot(rng_integers(self.rng, 2, size=(self.d, self.bits), dtype=self.dtype_i), 2 ** np.arange(self.bits, dtype=self.dtype_i))
    ltm = np.tril(rng_integers(self.rng, 2, size=(self.d, self.bits, self.bits), dtype=self.dtype_i))
    _cscramble(dim=self.d, bits=self.bits, ltm=ltm, sv=self._sv)