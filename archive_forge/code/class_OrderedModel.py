import warnings
from statsmodels.compat.pandas import Appender
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import stats
from statsmodels.base.model import (
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
class OrderedModel(GenericLikelihoodModel):
    """Ordinal Model based on logistic or normal distribution

    The parameterization corresponds to the proportional odds model in the
    logistic case.
    The model assumes that the endogenous variable is ordered but that the
    labels have no numeric interpretation besides the ordering.

    The model is based on a latent linear variable, where we observe only a
    discretization.

    y_latent = X beta + u

    The observed variable is defined by the interval

    y = {0 if y_latent <= cut_0
         1 of cut_0 < y_latent <= cut_1
         ...
         K if cut_K < y_latent

    The probability of observing y=k conditional on the explanatory variables
    X is given by

    prob(y = k | x) = Prob(cut_k < y_latent <= cut_k+1)
                    = Prob(cut_k - x beta < u <= cut_k+1 - x beta
                    = F(cut_k+1 - x beta) - F(cut_k - x beta)

    Where F is the cumulative distribution of u which is either the normal
    or the logistic distribution, but can be set to any other continuous
    distribution. We use standardized distributions to avoid identifiability
    problems.

    Parameters
    ----------
    endog : array_like
        Endogenous or dependent ordered categorical variable with k levels.
        Labels or values of endog will internally transformed to consecutive
        integers, 0, 1, 2, ...
        pd.Series with ordered Categorical as dtype should be preferred as it
        gives the order relation between the levels.
        If endog is not a pandas Categorical, then categories are
        sorted in lexicographic order (by numpy.unique).
    exog : array_like
        Exogenous, explanatory variables. This should not include an intercept.
        pd.DataFrame are also accepted.
        see Notes about constant when using formulas
    offset : array_like
        Offset is added to the linear prediction with coefficient equal to 1.
    distr : string 'probit' or 'logit', or a distribution instance
        The default is currently 'probit' which uses the normal distribution
        and corresponds to an ordered Probit model. The distribution is
        assumed to have the main methods of scipy.stats distributions, mainly
        cdf, pdf and ppf. The inverse cdf, ppf, is only use to calculate
        starting values.

    Notes
    -----
    Status: experimental, core results are verified, still subclasses
    `GenericLikelihoodModel` which will change in future versions.

    The parameterization of OrderedModel requires that there is no constant in
    the model, neither explicit nor implicit. The constant is equivalent to
    shifting all thresholds and is therefore not separately identified.

    Patsy's formula specification does not allow a design matrix without
    explicit or implicit constant if there are categorical variables (or maybe
    splines) among explanatory variables. As workaround, statsmodels removes an
    explicit intercept.

    Consequently, there are two valid cases to get a design matrix without
    intercept when using formulas:

    - specify a model without explicit and implicit intercept which is possible
      if there are only numerical variables in the model.
    - specify a model with an explicit intercept which statsmodels will remove.

    Models with an implicit intercept will be overparameterized, the parameter
    estimates will not be fully identified, cov_params will not be invertible
    and standard errors might contain nans. The computed results will be
    dominated by numerical imprecision coming mainly from convergence tolerance
    and numerical derivatives.

    The model will raise a ValueError if a remaining constant is detected.

    """
    _formula_max_endog = np.inf

    def __init__(self, endog, exog, offset=None, distr='probit', **kwds):
        if distr == 'probit':
            self.distr = stats.norm
        elif distr == 'logit':
            self.distr = stats.logistic
        else:
            self.distr = distr
        if offset is not None:
            offset = np.asarray(offset)
        self.offset = offset
        endog, labels, is_pandas = self._check_inputs(endog, exog)
        super().__init__(endog, exog, **kwds)
        k_levels = None
        if not is_pandas:
            if self.endog.ndim == 1:
                unique, index = np.unique(self.endog, return_inverse=True)
                self.endog = index
                labels = unique
                if np.isnan(labels).any():
                    msg = 'NaN in dependent variable detected. Missing values need to be removed.'
                    raise ValueError(msg)
            elif self.endog.ndim == 2:
                if not hasattr(self, 'design_info'):
                    raise ValueError('2-dim endog not supported')
                k_levels = self.endog.shape[1]
                labels = []
        if self.k_constant > 0:
            raise ValueError('There should not be a constant in the model')
        self._initialize_labels(labels, k_levels=k_levels)
        self.k_extra = self.k_levels - 1
        self.df_model = self.k_vars
        self.df_resid = self.nobs - (self.k_vars + self.k_extra)
        self.results_class = OrderedResults

    def _check_inputs(self, endog, exog):
        """Handle endog that is pandas Categorical.

        Checks if self.distrib is legal and provides Pandas ordered Categorical
        support for endog.

        Parameters
        ----------
        endog : array_like
            Endogenous, dependent variable, 1-D.
        exog : array_like
            Exogenous, explanatory variables.
            Currently not used.

        Returns
        -------
        endog : array_like or pandas Series
            If the original endog is a pandas ordered Categorical Series,
            then the returned endog are the ``codes``, i.e. integer
            representation of ordere categorical variable
        labels : None or list
            If original endog is pandas ordered Categorical Series, then the
            categories are returned. Otherwise ``labels`` is None.
        is_pandas : bool
            This is True if original endog is a pandas ordered Categorical
            Series and False otherwise.

        """
        if not isinstance(self.distr, stats.rv_continuous):
            msg = f'{self.distr.name} is not a scipy.stats distribution.'
            warnings.warn(msg)
        labels = None
        is_pandas = False
        if isinstance(endog, pd.Series):
            if isinstance(endog.dtypes, CategoricalDtype):
                if not endog.dtype.ordered:
                    warnings.warn('the endog has ordered == False, risk of capturing a wrong order for the categories. ordered == True preferred.', Warning)
                endog_name = endog.name
                labels = endog.values.categories
                endog = endog.cat.codes
                if endog.min() == -1:
                    raise ValueError('missing values in categorical endog are not supported')
                endog.name = endog_name
                is_pandas = True
        return (endog, labels, is_pandas)

    def _initialize_labels(self, labels, k_levels=None):
        self.labels = labels
        if k_levels is None:
            self.k_levels = len(labels)
        else:
            self.k_levels = k_levels
        if self.exog is not None:
            self.nobs, self.k_vars = self.exog.shape
        else:
            self.nobs, self.k_vars = (self.endog.shape[0], 0)
        threshold_names = [str(x) + '/' + str(y) for x, y in zip(labels[:-1], labels[1:])]
        if self.exog is not None:
            if len(self.exog_names) > self.k_vars:
                raise RuntimeError('something wrong with exog_names, too long')
            self.exog_names.extend(threshold_names)
        else:
            self.data.xnames = threshold_names

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None, *args, **kwargs):
        endog_name = formula.split('~')[0].strip()
        original_endog = data[endog_name]
        model = super().from_formula(formula, *args, data=data, drop_cols=['Intercept'], **kwargs)
        if model.endog.ndim == 2:
            if not (isinstance(original_endog.dtype, CategoricalDtype) and original_endog.dtype.ordered):
                msg = 'Only ordered pandas Categorical are supported as endog in formulas'
                raise ValueError(msg)
            labels = original_endog.values.categories
            model._initialize_labels(labels)
            model.endog = model.endog.argmax(1)
            model.data.ynames = endog_name
        return model
    from_formula.__func__.__doc__ = Model.from_formula.__doc__

    def cdf(self, x):
        """Cdf evaluated at x.

        Parameters
        ----------
        x : array_like
            Points at which cdf is evaluated. In the model `x` is the latent
            variable plus threshold constants.

        Returns
        -------
        Value of the cumulative distribution function of the underlying latent
        variable evaluated at x.
        """
        return self.distr.cdf(x)

    def pdf(self, x):
        """Pdf evaluated at x

        Parameters
        ----------
        x : array_like
            Points at which cdf is evaluated. In the model `x` is the latent
            variable plus threshold constants.

        Returns
        -------
        Value of the probability density function of the underlying latent
        variable evaluated at x.
        """
        return self.distr.pdf(x)

    def prob(self, low, upp):
        """Interval probability.

        Probability that value is in interval (low, upp], computed as

            prob = cdf(upp) - cdf(low)

        Parameters
        ----------
        low : array_like
            lower bound for interval
        upp : array_like
            upper bound for interval

        Returns
        -------
        float or ndarray
            Probability that value falls in interval (low, upp]

        """
        return np.maximum(self.cdf(upp) - self.cdf(low), 0)

    def transform_threshold_params(self, params):
        """transformation of the parameters in the optimization

        Parameters
        ----------
        params : nd_array
            Contains (exog_coef, transformed_thresholds) where exog_coef are
            the coefficient for the explanatory variables in the linear term,
            transformed threshold or cutoff points. The first, lowest threshold
            is unchanged, all other thresholds are in terms of exponentiated
            increments.

        Returns
        -------
        thresh : nd_array
            Thresh are the thresholds or cutoff constants for the intervals.

        """
        th_params = params[-(self.k_levels - 1):]
        thresh = np.concatenate((th_params[:1], np.exp(th_params[1:]))).cumsum()
        thresh = np.concatenate(([-np.inf], thresh, [np.inf]))
        return thresh

    def transform_reverse_threshold_params(self, params):
        """obtain transformed thresholds from original thresholds or cutoffs

        Parameters
        ----------
        params : ndarray
            Threshold values, cutoff constants for choice intervals, which
            need to be monotonically increasing.

        Returns
        -------
        thresh_params : ndarrray
            Transformed threshold parameter.
            The first, lowest threshold is unchanged, all other thresholds are
            in terms of exponentiated increments.
            Transformed parameters can be any real number without restrictions.

        """
        thresh_params = np.concatenate((params[:1], np.log(np.diff(params[:-1]))))
        return thresh_params

    def predict(self, params, exog=None, offset=None, which='prob'):
        """
        Predicted probabilities for each level of the ordinal endog.

        Parameters
        ----------
        params : ndarray
            Parameters for the Model, (exog_coef, transformed_thresholds).
        exog : array_like, optional
            Design / exogenous data. If exog is None, model exog is used.
        offset : array_like, optional
            Offset is added to the linear prediction with coefficient
            equal to 1. If offset is not provided and exog
            is None, uses the model's offset if present.  If not, uses
            0 as the default value.
        which : {"prob", "linpred", "cumprob"}
            Determines which statistic is predicted.

            - prob : predicted probabilities to be in each choice. 2-dim.
            - linear : 1-dim linear prediction of the latent variable
              ``x b + offset``
            - cumprob : predicted cumulative probability to be in choice k or
              lower

        Returns
        -------
        predicted values : ndarray
            If which is "prob", then 2-dim predicted probabilities with
            observations in rows and one column for each category or level of
            the categorical dependent variable.
            If which is "cumprob", then "prob" ar cumulatively added to get the
            cdf at k, i.e. probability of observing choice k or lower.
            If which is "linpred", then the conditional prediction of the
            latent variable is returned. In this case, the return is
            one-dimensional.
        """
        thresh = self.transform_threshold_params(params)
        xb = self._linpred(params, exog=exog, offset=offset)
        if which == 'linpred':
            return xb
        xb = xb[:, None]
        low = thresh[:-1] - xb
        upp = thresh[1:] - xb
        if which == 'prob':
            prob = self.prob(low, upp)
            return prob
        elif which in ['cum', 'cumprob']:
            cumprob = self.cdf(upp)
            return cumprob
        else:
            raise ValueError('`which` is not available')

    def _linpred(self, params, exog=None, offset=None):
        """Linear prediction of latent variable `x b + offset`.

        Parameters
        ----------
        params : ndarray
            Parameters for the model, (exog_coef, transformed_thresholds)
        exog : array_like, optional
            Design / exogenous data. Is exog is None, model exog is used.
        offset : array_like, optional
            Offset is added to the linear prediction with coefficient
            equal to 1. If offset is not provided and exog
            is None, uses the model's offset if present.  If not, uses
            0 as the default value.

        Returns
        -------
        linear : ndarray
            1-dim linear prediction given by exog times linear params plus
            offset. This is the prediction for the underlying latent variable.
            If exog and offset are None, then the predicted values are zero.

        """
        if exog is None:
            exog = self.exog
            if offset is None:
                offset = self.offset
        elif offset is None:
            offset = 0
        if offset is not None:
            offset = np.asarray(offset)
        if exog is not None:
            _exog = np.asarray(exog)
            _params = np.asarray(params)
            linpred = _exog.dot(_params[:-(self.k_levels - 1)])
        else:
            linpred = np.zeros(self.nobs)
        if offset is not None:
            linpred += offset
        return linpred

    def _bounds(self, params):
        """Integration bounds for the observation specific interval.

        This defines the lower and upper bounds for the intervals of the
        choices of all observations.

        The bounds for observation are given by

            a_{k_i-1} - linpred_i, a_k_i - linpred_i

        where
        - k_i is the choice in observation i.
        - a_{k_i-1} and a_k_i are thresholds (cutoffs) for choice k_i
        - linpred_i is the linear prediction for observation i

        Parameters
        ----------
        params : ndarray
            Parameters for the model, (exog_coef, transformed_thresholds)

        Return
        ------
        low : ndarray
            Lower bounds for choice intervals of each observation,
            1-dim with length nobs
        upp : ndarray
            Upper bounds for choice intervals of each observation,
            1-dim with length nobs.

        """
        thresh = self.transform_threshold_params(params)
        thresh_i_low = thresh[self.endog]
        thresh_i_upp = thresh[self.endog + 1]
        xb = self._linpred(params)
        low = thresh_i_low - xb
        upp = thresh_i_upp - xb
        return (low, upp)

    @Appender(GenericLikelihoodModel.loglike.__doc__)
    def loglike(self, params):
        return self.loglikeobs(params).sum()

    def loglikeobs(self, params):
        """
        Log-likelihood of OrderdModel for all observations.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike_obs : array_like
            The log likelihood for each observation of the model evaluated
            at ``params``.
        """
        low, upp = self._bounds(params)
        prob = self.prob(low, upp)
        return np.log(prob + 1e-20)

    def score_obs_(self, params):
        """score, first derivative of loglike for each observations

        This currently only implements the derivative with respect to the
        exog parameters, but not with respect to threshold parameters.

        """
        low, upp = self._bounds(params)
        prob = self.prob(low, upp)
        pdf_upp = self.pdf(upp)
        pdf_low = self.pdf(low)
        score_factor = (pdf_upp - pdf_low)[:, None]
        score_factor /= prob[:, None]
        so = np.column_stack((-score_factor[:, :1] * self.exog, score_factor[:, 1:]))
        return so

    @property
    def start_params(self):
        """Start parameters for the optimization corresponding to null model.

        The threshold are computed from the observed frequencies and
        transformed to the exponential increments parameterization.
        The parameters for explanatory variables are set to zero.
        """
        freq = np.bincount(self.endog) / len(self.endog)
        start_ppf = self.distr.ppf(np.clip(freq.cumsum(), 0, 1))
        start_threshold = self.transform_reverse_threshold_params(start_ppf)
        start_params = np.concatenate((np.zeros(self.k_vars), start_threshold))
        return start_params

    @Appender(LikelihoodModel.fit.__doc__)
    def fit(self, start_params=None, method='nm', maxiter=500, full_output=1, disp=1, callback=None, retall=0, **kwargs):
        fit_method = super().fit
        mlefit = fit_method(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, **kwargs)
        ordmlefit = OrderedResults(self, mlefit)
        ordmlefit.hasconst = 0
        result = OrderedResultsWrapper(ordmlefit)
        return result