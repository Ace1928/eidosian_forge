from statsmodels.compat.numpy import NP_LT_123
import numpy as np
from scipy.special import comb
from statsmodels.graphics.utils import _import_mpl
from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import itertools
from multiprocessing import Pool
from . import utils
def _band_quantiles(band, use_brute=use_brute, seed=seed):
    """
        Find extreme curves for a quantile band.

        From the `band` of quantiles, the associated PDF extrema values
        are computed. If `min_alpha` is not provided (single quantile value),
        `max_pdf` is set to `1E6` in order not to constrain the problem on high
        values.

        An optimization is performed per component in order to find the min and
        max curves. This is done by comparing the PDF value of a given curve
        with the band PDF.

        Parameters
        ----------
        band : array_like
            alpha values ``(max_alpha, min_alpha)`` ex: ``[0.9, 0.5]``
        use_brute : bool
            Use the brute force optimizer instead of the default differential
            evolution to find the curves. Default is False.
        seed : {None, int, np.random.RandomState}
            Seed value to pass to scipy.optimize.differential_evolution. Can
            be an integer or RandomState instance. If None, then the default
            RandomState provided by np.random is used.


        Returns
        -------
        band_quantiles : list of 1-D array
            ``(max_quantile, min_quantile)`` (2, n_features)
        """
    min_pdf = pvalues[alpha.index(band[0])]
    try:
        max_pdf = pvalues[alpha.index(band[1])]
    except IndexError:
        max_pdf = 1000000.0
    band = [min_pdf, max_pdf]
    pool = Pool()
    data = zip(range(dim), itertools.repeat((band, pca, bounds, ks_gaussian, seed, use_brute)))
    band_quantiles = pool.map(_min_max_band, data)
    pool.terminate()
    pool.close()
    band_quantiles = list(zip(*band_quantiles))
    return band_quantiles