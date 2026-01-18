import math
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from inspect import signature
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
from ..base import clone
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import pairwise_kernels
from ..utils.validation import _num_samples
def _check_bounds_params(self):
    """Called after fitting to warn if bounds may have been too tight."""
    list_close = np.isclose(self.bounds, np.atleast_2d(self.theta).T)
    idx = 0
    for hyp in self.hyperparameters:
        if hyp.fixed:
            continue
        for dim in range(hyp.n_elements):
            if list_close[idx, 0]:
                warnings.warn('The optimal value found for dimension %s of parameter %s is close to the specified lower bound %s. Decreasing the bound and calling fit again may find a better value.' % (dim, hyp.name, hyp.bounds[dim][0]), ConvergenceWarning)
            elif list_close[idx, 1]:
                warnings.warn('The optimal value found for dimension %s of parameter %s is close to the specified upper bound %s. Increasing the bound and calling fit again may find a better value.' % (dim, hyp.name, hyp.bounds[dim][1]), ConvergenceWarning)
            idx += 1