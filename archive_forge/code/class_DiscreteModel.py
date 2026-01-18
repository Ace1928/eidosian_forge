from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
class DiscreteModel(base.LikelihoodModel):
    """
    Abstract class for discrete choice models.

    This class does not do anything itself but lays out the methods and
    call signature expected of child classes in addition to those of
    statsmodels.model.LikelihoodModel.
    """

    def __init__(self, endog, exog, check_rank=True, **kwargs):
        self._check_rank = check_rank
        super().__init__(endog, exog, **kwargs)
        self.raise_on_perfect_prediction = False
        self.k_extra = 0

    def initialize(self):
        """
        Initialize is called by
        statsmodels.model.LikelihoodModel.__init__
        and should contain any preprocessing that needs to be done for a model.
        """
        if self._check_rank:
            rank = tools.matrix_rank(self.exog, method='qr')
        else:
            rank = self.exog.shape[1]
        self.df_model = float(rank - 1)
        self.df_resid = float(self.exog.shape[0] - rank)

    def cdf(self, X):
        """
        The cumulative distribution function of the model.
        """
        raise NotImplementedError

    def pdf(self, X):
        """
        The probability density (mass) function of the model.
        """
        raise NotImplementedError

    def _check_perfect_pred(self, params, *args):
        endog = self.endog
        fittedvalues = self.predict(params)
        if np.allclose(fittedvalues - endog, 0):
            if self.raise_on_perfect_prediction:
                msg = 'Perfect separation detected, results not available'
                raise PerfectSeparationError(msg)
            else:
                msg = 'Perfect separation or prediction detected, parameter may not be identified'
                warnings.warn(msg, category=PerfectSeparationWarning)

    @Appender(base.LikelihoodModel.fit.__doc__)
    def fit(self, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None, **kwargs):
        """
        Fit the model using maximum likelihood.

        The rest of the docstring is from
        statsmodels.base.model.LikelihoodModel.fit
        """
        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass
        mlefit = super().fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, **kwargs)
        return mlefit

    def fit_regularized(self, start_params=None, method='l1', maxiter='defined_by_method', full_output=1, disp=True, callback=None, alpha=0, trim_mode='auto', auto_trim_tol=0.01, size_trim_tol=0.0001, qc_tol=0.03, qc_verbose=False, **kwargs):
        """
        Fit the model using a regularized maximum likelihood.

        The regularization method AND the solver used is determined by the
        argument method.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : 'l1' or 'l1_cvxopt_cp'
            See notes for details.
        maxiter : {int, 'defined_by_method'}
            Maximum number of iterations to perform.
            If 'defined_by_method', then use method defaults (see notes).
        full_output : bool
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool
            Set to True to print convergence messages.
        fargs : tuple
            Extra arguments passed to the likelihood function, i.e.,
            loglike(x,*args).
        callback : callable callback(xk)
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        retall : bool
            Set to True to return list of solutions at each iteration.
            Available in Results object's mle_retvals attribute.
        alpha : non-negative scalar or numpy array (same size as parameters)
            The weight multiplying the l1 penalty term.
        trim_mode : 'auto, 'size', or 'off'
            If not 'off', trim (set to zero) parameters that would have been
            zero if the solver reached the theoretical minimum.
            If 'auto', trim params using the Theory above.
            If 'size', trim params if they have very small absolute value.
        size_trim_tol : float or 'auto' (default = 'auto')
            Tolerance used when trim_mode == 'size'.
        auto_trim_tol : float
            Tolerance used when trim_mode == 'auto'.
        qc_tol : float
            Print warning and do not allow auto trim when (ii) (above) is
            violated by this much.
        qc_verbose : bool
            If true, print out a full QC report upon failure.
        **kwargs
            Additional keyword arguments used when fitting the model.

        Returns
        -------
        Results
            A results instance.

        Notes
        -----
        Using 'l1_cvxopt_cp' requires the cvxopt module.

        Extra parameters are not penalized if alpha is given as a scalar.
        An example is the shape parameter in NegativeBinomial `nb1` and `nb2`.

        Optional arguments for the solvers (available in Results.mle_settings)::

            'l1'
                acc : float (default 1e-6)
                    Requested accuracy as used by slsqp
            'l1_cvxopt_cp'
                abstol : float
                    absolute accuracy (default: 1e-7).
                reltol : float
                    relative accuracy (default: 1e-6).
                feastol : float
                    tolerance for feasibility conditions (default: 1e-7).
                refinement : int
                    number of iterative refinement steps when solving KKT
                    equations (default: 1).

        Optimization methodology

        With :math:`L` the negative log likelihood, we solve the convex but
        non-smooth problem

        .. math:: \\min_\\beta L(\\beta) + \\sum_k\\alpha_k |\\beta_k|

        via the transformation to the smooth, convex, constrained problem
        in twice as many variables (adding the "added variables" :math:`u_k`)

        .. math:: \\min_{\\beta,u} L(\\beta) + \\sum_k\\alpha_k u_k,

        subject to

        .. math:: -u_k \\leq \\beta_k \\leq u_k.

        With :math:`\\partial_k L` the derivative of :math:`L` in the
        :math:`k^{th}` parameter direction, theory dictates that, at the
        minimum, exactly one of two conditions holds:

        (i) :math:`|\\partial_k L| = \\alpha_k`  and  :math:`\\beta_k \\neq 0`
        (ii) :math:`|\\partial_k L| \\leq \\alpha_k`  and  :math:`\\beta_k = 0`
        """
        _validate_l1_method(method)
        cov_params_func = self.cov_params_func_l1
        alpha = np.array(alpha)
        assert alpha.min() >= 0
        try:
            kwargs['alpha'] = alpha
        except TypeError:
            kwargs = dict(alpha=alpha)
        kwargs['alpha_rescaled'] = kwargs['alpha'] / float(self.endog.shape[0])
        kwargs['trim_mode'] = trim_mode
        kwargs['size_trim_tol'] = size_trim_tol
        kwargs['auto_trim_tol'] = auto_trim_tol
        kwargs['qc_tol'] = qc_tol
        kwargs['qc_verbose'] = qc_verbose
        if maxiter == 'defined_by_method':
            if method == 'l1':
                maxiter = 1000
            elif method == 'l1_cvxopt_cp':
                maxiter = 70
        extra_fit_funcs = {'l1': fit_l1_slsqp}
        if have_cvxopt and method == 'l1_cvxopt_cp':
            from statsmodels.base.l1_cvxopt import fit_l1_cvxopt_cp
            extra_fit_funcs['l1_cvxopt_cp'] = fit_l1_cvxopt_cp
        elif method.lower() == 'l1_cvxopt_cp':
            raise ValueError("Cannot use l1_cvxopt_cp as cvxopt was not found (install it, or use method='l1' instead)")
        if callback is None:
            callback = self._check_perfect_pred
        else:
            pass
        mlefit = super().fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback, extra_fit_funcs=extra_fit_funcs, cov_params_func=cov_params_func, **kwargs)
        return mlefit

    def cov_params_func_l1(self, likelihood_model, xopt, retvals):
        """
        Computes cov_params on a reduced parameter space
        corresponding to the nonzero parameters resulting from the
        l1 regularized fit.

        Returns a full cov_params matrix, with entries corresponding
        to zero'd values set to np.nan.
        """
        H = likelihood_model.hessian(xopt)
        trimmed = retvals['trimmed']
        nz_idx = np.nonzero(~trimmed)[0]
        nnz_params = (~trimmed).sum()
        if nnz_params > 0:
            H_restricted = H[nz_idx[:, None], nz_idx]
            H_restricted_inv = np.linalg.inv(-H_restricted)
        else:
            H_restricted_inv = np.zeros(0)
        cov_params = np.nan * np.ones(H.shape)
        cov_params[nz_idx[:, None], nz_idx] = H_restricted_inv
        return cov_params

    def predict(self, params, exog=None, which='mean', linear=None):
        """
        Predict response variable of a model given exogenous variables.
        """
        raise NotImplementedError

    def _derivative_exog(self, params, exog=None, dummy_idx=None, count_idx=None):
        """
        This should implement the derivative of the non-linear function
        """
        raise NotImplementedError

    def _derivative_exog_helper(self, margeff, params, exog, dummy_idx, count_idx, transform):
        """
        Helper for _derivative_exog to wrap results appropriately
        """
        from .discrete_margins import _get_count_effects, _get_dummy_effects
        if count_idx is not None:
            margeff = _get_count_effects(margeff, exog, count_idx, transform, self, params)
        if dummy_idx is not None:
            margeff = _get_dummy_effects(margeff, exog, dummy_idx, transform, self, params)
        return margeff