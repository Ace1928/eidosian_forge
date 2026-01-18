import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2, norm
from statsmodels.graphics import utils
class CumIncidenceRight:
    """
    Estimation and inference for a cumulative incidence function.

    If J = 1, 2, ... indicates the event type, the cumulative
    incidence function for cause j is:

    I(t, j) = P(T <= t and J=j)

    Only right censoring is supported.  If frequency weights are provided,
    the point estimate is returned without a standard error.

    Parameters
    ----------
    time : array_like
        An array of times (censoring times or event times)
    status : array_like
        If status >= 1 indicates which event occurred at time t.  If
        status = 0, the subject was censored at time t.
    title : str
        Optional title used for plots and summary output.
    freq_weights : array_like
        Optional frequency weights
    exog : array_like
        Optional, if present used to account for violation of
        independent censoring.
    bw_factor : float
        Band-width multiplier for kernel-based estimation.  Only
        used if exog is provided.
    dimred : bool
        If True, proportional hazards regression models are used to
        reduce exog to two columns by predicting overall events and
        censoring in two separate models.  If False, exog is used
        directly for calculating kernel weights without dimension
        reduction.

    Attributes
    ----------
    times : array_like
        The distinct times at which the incidence rates are estimated
    cinc : list of arrays
        cinc[k-1] contains the estimated cumulative incidence rates
        for outcome k=1,2,...
    cinc_se : list of arrays
        The standard errors for the values in `cinc`.  Not available when
        exog and/or frequency weights are provided.

    Notes
    -----
    When exog is provided, a local estimate of the cumulative incidence
    rate around each point is provided, and these are averaged to
    produce an estimate of the marginal cumulative incidence
    functions.  The procedure is analogous to that described in Zeng
    (2004) for estimation of the marginal survival function.  The
    approach removes bias resulting from dependent censoring when the
    censoring becomes independent conditioned on the columns of exog.

    References
    ----------
    The Stata stcompet procedure:
        http://www.stata-journal.com/sjpdf.html?articlenum=st0059

    Dinse, G. E. and M. G. Larson. 1986. A note on semi-Markov models
    for partially censored data. Biometrika 73: 379-386.

    Marubini, E. and M. G. Valsecchi. 1995. Analysing Survival Data
    from Clinical Trials and Observational Studies. Chichester, UK:
    John Wiley & Sons.

    D. Zeng (2004).  Estimating marginal survival function by
    adjusting for dependent censoring using many covariates.  Annals
    of Statistics 32:4.
    https://arxiv.org/pdf/math/0409180.pdf
    """

    def __init__(self, time, status, title=None, freq_weights=None, exog=None, bw_factor=1.0, dimred=True):
        _checkargs(time, status, None, freq_weights, None)
        time = self.time = np.asarray(time)
        status = self.status = np.asarray(status)
        if freq_weights is not None:
            freq_weights = self.freq_weights = np.asarray(freq_weights)
        if exog is not None:
            from ._kernel_estimates import _kernel_cumincidence
            exog = self.exog = np.asarray(exog)
            nobs = exog.shape[0]
            kw = nobs ** (-1 / 3.0) * bw_factor
            kfunc = lambda x: np.exp(-x ** 2 / kw ** 2).sum(1)
            x = _kernel_cumincidence(time, status, exog, kfunc, freq_weights, dimred)
            self.times = x[0]
            self.cinc = x[1]
            return
        x = _calc_incidence_right(time, status, freq_weights)
        self.cinc = x[0]
        self.cinc_se = x[1]
        self.times = x[2]
        self.title = '' if not title else title