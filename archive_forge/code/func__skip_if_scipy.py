from random import random
from typing import Callable
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag
import cirq
from cirq.transformers.analytical_decompositions.three_qubit_decomposition import (
def _skip_if_scipy(*, version_is_greater_than_1_5_0: bool) -> Callable[[Callable], Callable]:

    def decorator(func):
        try:
            from scipy.linalg import cossin
            return None if version_is_greater_than_1_5_0 else func
        except ImportError:
            return func if version_is_greater_than_1_5_0 else None
    return decorator