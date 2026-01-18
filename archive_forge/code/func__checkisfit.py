import numpy as np
from scipy import integrate, stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.validation import array_like, float_like
from . import bandwidths
from .kdetools import forrt, revrt, silverman_transform
from .linbin import fast_linbin
def _checkisfit(self):
    try:
        self.density
    except Exception:
        raise ValueError('Call fit to fit the density first')