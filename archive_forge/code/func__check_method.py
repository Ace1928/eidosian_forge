from __future__ import annotations
from typing import Any
from collections.abc import Sequence
import numpy as np
from scipy import optimize
from statsmodels.compat.scipy import SP_LT_15, SP_LT_17
def _check_method(method, methods):
    if method not in methods:
        message = 'Unknown fit method %s' % method
        raise ValueError(message)