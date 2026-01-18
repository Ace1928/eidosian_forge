import math
from numbers import Real
import numpy as np
from scipy.special import betaln, digamma, gammaln
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import (
def _check_weights_parameters(self):
    """Check the parameter of the Dirichlet distribution."""
    if self.weight_concentration_prior is None:
        self.weight_concentration_prior_ = 1.0 / self.n_components
    else:
        self.weight_concentration_prior_ = self.weight_concentration_prior