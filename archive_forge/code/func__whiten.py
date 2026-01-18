from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
def _whiten(self, x):
    return x @ self._LP