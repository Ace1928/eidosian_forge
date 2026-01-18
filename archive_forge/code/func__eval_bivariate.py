from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
def _eval_bivariate(self, x1, x2, weights):
    """Inner function for ECDF of two variables."""
    raise NotImplementedError('Bivariate ECDF is not implemented')