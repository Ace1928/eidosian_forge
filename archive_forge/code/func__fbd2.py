from statsmodels.compat.numpy import NP_LT_123
import numpy as np
from scipy.special import comb
from statsmodels.graphics.utils import _import_mpl
from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import itertools
from multiprocessing import Pool
from . import utils
def _fbd2():
    down = np.min(rmat, axis=1) - 1
    up = n - np.max(rmat, axis=1)
    return (up * down + n - 1) / comb(n, 2)