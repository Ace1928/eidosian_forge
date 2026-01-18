from statsmodels.compat.numpy import NP_LT_123
import numpy as np
from scipy.special import comb
from statsmodels.graphics.utils import _import_mpl
from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import itertools
from multiprocessing import Pool
from . import utils
def _min_max_band(args):
    """
    Min and max values at `idx`.

    Global optimization to find the extrema per component.

    Parameters
    ----------
    args: list
        It is a list of an idx and other arguments as a tuple:
            idx : int
                Index value of the components to compute
        The tuple contains:
            band : list of float
                PDF values `[min_pdf, max_pdf]` to be within.
            pca : statsmodels Principal Component Analysis instance
                The PCA object to use.
            bounds : sequence
                ``(min, max)`` pair for each components
            ks_gaussian : KDEMultivariate instance

    Returns
    -------
    band : tuple of float
        ``(max, min)`` curve values at `idx`
    """
    idx, (band, pca, bounds, ks_gaussian, use_brute, seed) = args
    if have_de_optim and (not use_brute):
        max_ = differential_evolution(_curve_constrained, bounds=bounds, args=(idx, -1, band, pca, ks_gaussian), maxiter=7, seed=seed).x
        min_ = differential_evolution(_curve_constrained, bounds=bounds, args=(idx, 1, band, pca, ks_gaussian), maxiter=7, seed=seed).x
    else:
        max_ = brute(_curve_constrained, ranges=bounds, finish=fmin, args=(idx, -1, band, pca, ks_gaussian))
        min_ = brute(_curve_constrained, ranges=bounds, finish=fmin, args=(idx, 1, band, pca, ks_gaussian))
    band = (_inverse_transform(pca, max_)[0][idx], _inverse_transform(pca, min_)[0][idx])
    return band