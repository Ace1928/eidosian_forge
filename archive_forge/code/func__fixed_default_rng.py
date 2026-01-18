import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
import numpy as np
from scipy._lib._array_api import array_namespace
@contextmanager
def _fixed_default_rng(seed=1638083107694713882823079058616272161):
    """Context with a fixed np.random.default_rng seed."""
    orig_fun = np.random.default_rng
    np.random.default_rng = lambda seed=seed: orig_fun(seed)
    try:
        yield
    finally:
        np.random.default_rng = orig_fun