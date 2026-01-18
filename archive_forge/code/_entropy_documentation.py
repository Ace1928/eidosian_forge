from __future__ import annotations
import math
import numpy as np
from scipy import special
from ._axis_nan_policy import _axis_nan_policy_factory, _broadcast_arrays
Compute the Correa estimator as described in [6].