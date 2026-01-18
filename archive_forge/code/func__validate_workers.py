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
def _validate_workers(workers: IntNumber=1) -> IntNumber:
    """Validate `workers` based on platform and value.

    Parameters
    ----------
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is
        given all CPU threads are used. Default is 1.

    Returns
    -------
    Workers : int
        Number of CPU used by the algorithm

    """
    workers = int(workers)
    if workers == -1:
        workers = os.cpu_count()
        if workers is None:
            raise NotImplementedError('Cannot determine the number of cpus using os.cpu_count(), cannot use -1 for the number of workers')
    elif workers <= 0:
        raise ValueError(f'Invalid number of workers: {workers}, must be -1 or > 0')
    return workers