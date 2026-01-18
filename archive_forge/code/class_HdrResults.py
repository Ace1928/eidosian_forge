from statsmodels.compat.numpy import NP_LT_123
import numpy as np
from scipy.special import comb
from statsmodels.graphics.utils import _import_mpl
from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import itertools
from multiprocessing import Pool
from . import utils
class HdrResults:
    """Wrap results and pretty print them."""

    def __init__(self, kwds):
        self.__dict__.update(kwds)

    def __repr__(self):
        msg = 'HDR boxplot summary:\n-> median:\n{}\n-> 50% HDR (max, min):\n{}\n-> 90% HDR (max, min):\n{}\n-> Extra quantiles (max, min):\n{}\n-> Outliers:\n{}\n-> Outliers indices:\n{}\n'.format(self.median, self.hdr_50, self.hdr_90, self.extra_quantiles, self.outliers, self.outliers_idx)
        return msg