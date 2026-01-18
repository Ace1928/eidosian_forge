from __future__ import annotations
import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
import inspect
from scipy._lib._util import check_random_state, _rename_parameter
from scipy.special import ndtr, ndtri, comb, factorial
from scipy._lib._util import rng_integers
from dataclasses import dataclass
from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning
def _batch_generator(iterable, batch):
    """A generator that yields batches of elements from an iterable"""
    iterator = iter(iterable)
    if batch <= 0:
        raise ValueError('`batch` must be positive.')
    z = [item for i, item in zip(range(batch), iterator)]
    while z:
        yield z
        z = [item for i, item in zip(range(batch), iterator)]