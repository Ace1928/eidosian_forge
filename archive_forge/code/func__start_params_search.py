import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
def _start_params_search(self, reps, start_params=None, transformed=True, em_iter=5, scale=1.0):
    """
        Search for starting parameters as random permutations of a vector

        Parameters
        ----------
        reps : int
            Number of random permutations to try.
        start_params : ndarray, optional
            Starting parameter vector. If not given, class-level start
            parameters are used.
        transformed : bool, optional
            If `start_params` was provided, whether or not those parameters
            are already transformed. Default is True.
        em_iter : int, optional
            Number of EM iterations to apply to each random permutation.
        scale : array or float, optional
            Scale of variates for random start parameter search. Can be given
            as an array of length equal to the number of parameters or as a
            single scalar.

        Notes
        -----
        This is a private method for finding good starting parameters for MLE
        by scoring, where the defaults have been set heuristically.
        """
    if start_params is None:
        start_params = self.start_params
        transformed = True
    else:
        start_params = np.array(start_params, ndmin=1)
    if transformed:
        start_params = self.untransform_params(start_params)
    scale = np.array(scale, ndmin=1)
    if scale.size == 1:
        scale = np.ones(self.k_params) * scale
    if not scale.size == self.k_params:
        raise ValueError('Scale of variates for random start parameter search must be given for each parameter or as a single scalar.')
    variates = np.zeros((reps, self.k_params))
    for i in range(self.k_params):
        variates[:, i] = scale[i] * np.random.uniform(-0.5, 0.5, size=reps)
    llf = self.loglike(start_params, transformed=False)
    params = start_params
    for i in range(reps):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                proposed_params = self._fit_em(start_params + variates[i], transformed=False, maxiter=em_iter, return_params=True)
                proposed_llf = self.loglike(proposed_params)
                if proposed_llf > llf:
                    llf = proposed_llf
                    params = self.untransform_params(proposed_params)
            except Exception:
                pass
    return self.transform_params(params)