from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
def _deterministic_to_exog(deterministic, seasons, nobs_tot, first_season=0, seasons_centered=False, exog=None, exog_coint=None):
    """
    Translate all information about deterministic terms into a single array.

    These information is taken from `deterministic` and `seasons` as well as
    from the `exog` and `exog_coint` arrays. The resulting array form can then
    be used e.g. in VAR's __init__ method.

    Parameters
    ----------
    deterministic : str
        A string specifying the deterministic terms in the model. See VECM's
        docstring for more information.
    seasons : int
        Number of periods in a seasonal cycle.
    nobs_tot : int
        Number of observations including the presample.
    first_season : int, default: 0
        Season of the first observation.
    seasons_centered : bool, default: False
        If True, the seasonal dummy variables are demeaned such that they are
        orthogonal to an intercept term.
    exog : ndarray (nobs_tot x #det_terms) or None, default: None
        An ndarray representing deterministic terms outside the cointegration
        relation.
    exog_coint : ndarray (nobs_tot x #det_terms_coint) or None, default: None
        An ndarray representing deterministic terms inside the cointegration
        relation.

    Returns
    -------
    exog : ndarray or None
        None, if the function's arguments do not contain deterministic terms.
        Otherwise, an ndarray representing these deterministic terms.
    """
    exogs = []
    deterministic = string_like(deterministic, 'deterministic')
    if 'co' in deterministic or 'ci' in deterministic:
        exogs.append(np.ones(nobs_tot))
    if exog_coint is not None:
        exogs.append(exog_coint)
    if 'lo' in deterministic or 'li' in deterministic:
        exogs.append(np.arange(nobs_tot))
    if seasons > 0:
        exogs.append(seasonal_dummies(seasons, nobs_tot, first_period=first_season, centered=seasons_centered))
    if exog is not None:
        exogs.append(exog)
    return np.column_stack(exogs) if exogs else None