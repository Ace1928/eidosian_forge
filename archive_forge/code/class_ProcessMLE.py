import numpy as np
import pandas as pd
import patsy
import statsmodels.base.model as base
from statsmodels.regression.linear_model import OLS
import collections
from scipy.optimize import minimize
from statsmodels.iolib import summary2
from statsmodels.tools.numdiff import approx_fprime
import warnings
class ProcessMLE(base.LikelihoodModel):
    """
    Fit a Gaussian mean/variance regression model.

    This class fits a one-dimensional Gaussian process model with
    parametrized mean and covariance structures to grouped data.  For
    each group, there is an independent realization of a latent
    Gaussian process indexed by an observed real-valued time
    variable..  The data consist of the Gaussian process observed at a
    finite number of `time` values.

    The process mean and variance can be lined to covariates.  The
    mean structure is linear in the covariates.  The covariance
    structure is non-stationary, and is defined parametrically through
    'scaling', and 'smoothing' parameters.  The covariance of the
    process between two observations in the same group is a function
    of the distance between the time values of the two observations.
    The scaling and smoothing parameters can be linked to covariates.

    The observed data are modeled as the sum of the Gaussian process
    realization and (optionally) independent white noise.  The standard
    deviation of the white noise can be linked to covariates.

    The data should be provided in 'long form', with a group label to
    indicate which observations belong to the same group.
    Observations in different groups are always independent.

    Parameters
    ----------
    endog : array_like
        The dependent variable.
    exog : array_like
        The design matrix for the mean structure
    exog_scale : array_like
        The design matrix for the scaling structure
    exog_smooth : array_like
        The design matrix for the smoothness structure
    exog_noise : array_like
        The design matrix for the additive white noise. The
        linear predictor is the log of the white noise standard
        deviation.  If None, there is no additive noise (the
        process is observed directly).
    time : array_like (1-dimensional)
        The univariate index values, used to calculate distances
        between observations in the same group, which determines
        their correlations.
    groups : array_like (1-dimensional)
        The group values.
    cov : a ProcessCovariance instance
        Defaults to GaussianCovariance.
    """

    def __init__(self, endog, exog, exog_scale, exog_smooth, exog_noise, time, groups, cov=None, **kwargs):
        super().__init__(endog, exog, exog_scale=exog_scale, exog_smooth=exog_smooth, exog_noise=exog_noise, time=time, groups=groups, **kwargs)
        self._has_noise = exog_noise is not None
        xnames = []
        if hasattr(exog, 'columns'):
            xnames = list(exog.columns)
        else:
            xnames = ['Mean%d' % j for j in range(exog.shape[1])]
        if hasattr(exog_scale, 'columns'):
            xnames += list(exog_scale.columns)
        else:
            xnames += ['Scale%d' % j for j in range(exog_scale.shape[1])]
        if hasattr(exog_smooth, 'columns'):
            xnames += list(exog_smooth.columns)
        else:
            xnames += ['Smooth%d' % j for j in range(exog_smooth.shape[1])]
        if self._has_noise:
            if hasattr(exog_noise, 'columns'):
                xnames += list(exog_noise.columns)
            else:
                xnames += ['Noise%d' % j for j in range(exog_noise.shape[1])]
        self.data.param_names = xnames
        if cov is None:
            cov = GaussianCovariance()
        self.cov = cov
        _check_args(endog, exog, exog_scale, exog_smooth, exog_noise, time, groups)
        groups_ix = collections.defaultdict(list)
        for i, g in enumerate(groups):
            groups_ix[g].append(i)
        self._groups_ix = groups_ix
        self.verbose = False
        self.k_exog = self.exog.shape[1]
        self.k_scale = self.exog_scale.shape[1]
        self.k_smooth = self.exog_smooth.shape[1]
        if self._has_noise:
            self.k_noise = self.exog_noise.shape[1]

    def _split_param_names(self):
        xnames = self.data.param_names
        q = 0
        mean_names = xnames[q:q + self.k_exog]
        q += self.k_exog
        scale_names = xnames[q:q + self.k_scale]
        q += self.k_scale
        smooth_names = xnames[q:q + self.k_smooth]
        if self._has_noise:
            q += self.k_noise
            noise_names = xnames[q:q + self.k_noise]
        else:
            noise_names = []
        return (mean_names, scale_names, smooth_names, noise_names)

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None, *args, **kwargs):
        if 'scale_formula' in kwargs:
            scale_formula = kwargs['scale_formula']
        else:
            raise ValueError('scale_formula is a required argument')
        if 'smooth_formula' in kwargs:
            smooth_formula = kwargs['smooth_formula']
        else:
            raise ValueError('smooth_formula is a required argument')
        if 'noise_formula' in kwargs:
            noise_formula = kwargs['noise_formula']
        else:
            noise_formula = None
        if 'time' in kwargs:
            time = kwargs['time']
        else:
            raise ValueError('time is a required argument')
        if 'groups' in kwargs:
            groups = kwargs['groups']
        else:
            raise ValueError('groups is a required argument')
        if subset is not None:
            warnings.warn("'subset' is ignored")
        if drop_cols is not None:
            warnings.warn("'drop_cols' is ignored")
        if isinstance(time, str):
            time = np.asarray(data[time])
        if isinstance(groups, str):
            groups = np.asarray(data[groups])
        exog_scale = patsy.dmatrix(scale_formula, data)
        scale_design_info = exog_scale.design_info
        scale_names = scale_design_info.column_names
        exog_scale = np.asarray(exog_scale)
        exog_smooth = patsy.dmatrix(smooth_formula, data)
        smooth_design_info = exog_smooth.design_info
        smooth_names = smooth_design_info.column_names
        exog_smooth = np.asarray(exog_smooth)
        if noise_formula is not None:
            exog_noise = patsy.dmatrix(noise_formula, data)
            noise_design_info = exog_noise.design_info
            noise_names = noise_design_info.column_names
            exog_noise = np.asarray(exog_noise)
        else:
            exog_noise, noise_design_info, noise_names, exog_noise = (None, None, [], None)
        mod = super().from_formula(formula, data=data, subset=None, exog_scale=exog_scale, exog_smooth=exog_smooth, exog_noise=exog_noise, time=time, groups=groups)
        mod.data.scale_design_info = scale_design_info
        mod.data.smooth_design_info = smooth_design_info
        if mod._has_noise:
            mod.data.noise_design_info = noise_design_info
        mod.data.param_names = mod.exog_names + scale_names + smooth_names + noise_names
        return mod

    def unpack(self, z):
        """
        Split the packed parameter vector into blocks.
        """
        pm = self.exog.shape[1]
        mnpar = z[0:pm]
        pv = self.exog_scale.shape[1]
        scpar = z[pm:pm + pv]
        ps = self.exog_smooth.shape[1]
        smpar = z[pm + pv:pm + pv + ps]
        nopar = z[pm + pv + ps:]
        return (mnpar, scpar, smpar, nopar)

    def _get_start(self):
        model = OLS(self.endog, self.exog)
        result = model.fit()
        m = self.exog_scale.shape[1] + self.exog_smooth.shape[1]
        if self._has_noise:
            m += self.exog_noise.shape[1]
        return np.concatenate((result.params, np.zeros(m)))

    def loglike(self, params):
        """
        Calculate the log-likelihood function for the model.

        Parameters
        ----------
        params : array_like
            The packed parameters for the model.

        Returns
        -------
        The log-likelihood value at the given parameter point.

        Notes
        -----
        The mean, scaling, and smoothing parameters are packed into
        a vector.  Use `unpack` to access the component vectors.
        """
        mnpar, scpar, smpar, nopar = self.unpack(params)
        resid = self.endog - np.dot(self.exog, mnpar)
        sc = np.exp(np.dot(self.exog_scale, scpar))
        sm = np.exp(np.dot(self.exog_smooth, smpar))
        if self._has_noise:
            no = np.exp(np.dot(self.exog_noise, nopar))
        ll = 0.0
        for _, ix in self._groups_ix.items():
            cm = self.cov.get_cov(self.time[ix], sc[ix], sm[ix])
            if self._has_noise:
                cm.flat[::cm.shape[0] + 1] += no[ix] ** 2
            re = resid[ix]
            ll -= 0.5 * np.linalg.slogdet(cm)[1]
            ll -= 0.5 * np.dot(re, np.linalg.solve(cm, re))
        if self.verbose:
            print('L=', ll)
        return ll

    def score(self, params):
        """
        Calculate the score function for the model.

        Parameters
        ----------
        params : array_like
            The packed parameters for the model.

        Returns
        -------
        The score vector at the given parameter point.

        Notes
        -----
        The mean, scaling, and smoothing parameters are packed into
        a vector.  Use `unpack` to access the component vectors.
        """
        mnpar, scpar, smpar, nopar = self.unpack(params)
        pm, pv, ps = (len(mnpar), len(scpar), len(smpar))
        resid = self.endog - np.dot(self.exog, mnpar)
        sc = np.exp(np.dot(self.exog_scale, scpar))
        sm = np.exp(np.dot(self.exog_smooth, smpar))
        if self._has_noise:
            no = np.exp(np.dot(self.exog_noise, nopar))
        score = np.zeros(len(mnpar) + len(scpar) + len(smpar) + len(nopar))
        for _, ix in self._groups_ix.items():
            sc_i = sc[ix]
            sm_i = sm[ix]
            resid_i = resid[ix]
            time_i = self.time[ix]
            exog_i = self.exog[ix, :]
            exog_scale_i = self.exog_scale[ix, :]
            exog_smooth_i = self.exog_smooth[ix, :]
            cm = self.cov.get_cov(time_i, sc_i, sm_i)
            if self._has_noise:
                no_i = no[ix]
                exog_noise_i = self.exog_noise[ix, :]
                cm.flat[::cm.shape[0] + 1] += no[ix] ** 2
            cmi = np.linalg.inv(cm)
            jacv, jacs = self.cov.jac(time_i, sc_i, sm_i)
            dcr = np.linalg.solve(cm, resid_i)
            score[0:pm] += np.dot(exog_i.T, dcr)
            rx = np.outer(resid_i, resid_i)
            qm = np.linalg.solve(cm, rx)
            qm = 0.5 * np.linalg.solve(cm, qm.T)
            scx = sc_i[:, None] * exog_scale_i
            for i, _ in enumerate(ix):
                jq = np.sum(jacv[i] * qm)
                score[pm:pm + pv] += jq * scx[i, :]
                score[pm:pm + pv] -= 0.5 * np.sum(jacv[i] * cmi) * scx[i, :]
            smx = sm_i[:, None] * exog_smooth_i
            for i, _ in enumerate(ix):
                jq = np.sum(jacs[i] * qm)
                score[pm + pv:pm + pv + ps] += jq * smx[i, :]
                score[pm + pv:pm + pv + ps] -= 0.5 * np.sum(jacs[i] * cmi) * smx[i, :]
            if self._has_noise:
                sno = no_i[:, None] ** 2 * exog_noise_i
                score[pm + pv + ps:] -= np.dot(cmi.flat[::cm.shape[0] + 1], sno)
                bm = np.dot(cmi, np.dot(rx, cmi))
                score[pm + pv + ps:] += np.dot(bm.flat[::bm.shape[0] + 1], sno)
        if self.verbose:
            print('|G|=', np.sqrt(np.sum(score * score)))
        return score

    def hessian(self, params):
        hess = approx_fprime(params, self.score)
        return hess

    def fit(self, start_params=None, method=None, maxiter=None, **kwargs):
        """
        Fit a grouped Gaussian process regression using MLE.

        Parameters
        ----------
        start_params : array_like
            Optional starting values.
        method : str or array of str
            Method or sequence of methods for scipy optimize.
        maxiter : int
            The maximum number of iterations in the optimization.

        Returns
        -------
        An instance of ProcessMLEResults.
        """
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        minim_opts = {}
        if 'minim_opts' in kwargs:
            minim_opts = kwargs['minim_opts']
        if start_params is None:
            start_params = self._get_start()
        if isinstance(method, str):
            method = [method]
        elif method is None:
            method = ['powell', 'bfgs']
        for j, meth in enumerate(method):
            if meth not in ('powell',):

                def jac(x):
                    return -self.score(x)
            else:
                jac = None
            if maxiter is not None:
                if np.isscalar(maxiter):
                    minim_opts['maxiter'] = maxiter
                else:
                    minim_opts['maxiter'] = maxiter[j % len(maxiter)]
            f = minimize(lambda x: -self.loglike(x), method=meth, x0=start_params, jac=jac, options=minim_opts)
            if not f.success:
                msg = 'Fitting did not converge'
                if jac is not None:
                    msg += ', |gradient|=%.6f' % np.sqrt(np.sum(f.jac ** 2))
                if j < len(method) - 1:
                    msg += ', trying %s next...' % method[j + 1]
                warnings.warn(msg)
            if np.isfinite(f.x).all():
                start_params = f.x
        hess = self.hessian(f.x)
        try:
            cov_params = -np.linalg.inv(hess)
        except Exception:
            cov_params = None

        class rslt:
            pass
        r = rslt()
        r.params = f.x
        r.normalized_cov_params = cov_params
        r.optim_retvals = f
        r.scale = 1
        rslt = ProcessMLEResults(self, r)
        return rslt

    def covariance(self, time, scale_params, smooth_params, scale_data, smooth_data):
        """
        Returns a Gaussian process covariance matrix.

        Parameters
        ----------
        time : array_like
            The time points at which the fitted covariance matrix is
            calculated.
        scale_params : array_like
            The regression parameters for the scaling part
            of the covariance structure.
        smooth_params : array_like
            The regression parameters for the smoothing part
            of the covariance structure.
        scale_data : DataFrame
            The data used to determine the scale parameter,
            must have len(time) rows.
        smooth_data : DataFrame
            The data used to determine the smoothness parameter,
            must have len(time) rows.

        Returns
        -------
        A covariance matrix.

        Notes
        -----
        If the model was fit using formulas, `scale` and `smooth` should
        be Dataframes, containing all variables that were present in the
        respective scaling and smoothing formulas used to fit the model.
        Otherwise, `scale` and `smooth` should contain data arrays whose
        columns align with the fitted scaling and smoothing parameters.

        The covariance is only for the Gaussian process and does not include
        the white noise variance.
        """
        if not hasattr(self.data, 'scale_design_info'):
            sca = np.dot(scale_data, scale_params)
            smo = np.dot(smooth_data, smooth_params)
        else:
            sc = patsy.dmatrix(self.data.scale_design_info, scale_data)
            sm = patsy.dmatrix(self.data.smooth_design_info, smooth_data)
            sca = np.exp(np.dot(sc, scale_params))
            smo = np.exp(np.dot(sm, smooth_params))
        return self.cov.get_cov(time, sca, smo)

    def predict(self, params, exog=None, *args, **kwargs):
        """
        Obtain predictions of the mean structure.

        Parameters
        ----------
        params : array_like
            The model parameters, may be truncated to include only mean
            parameters.
        exog : array_like
            The design matrix for the mean structure.  If not provided,
            the model's design matrix is used.
        """
        if exog is None:
            exog = self.exog
        elif hasattr(self.data, 'design_info'):
            exog = patsy.dmatrix(self.data.design_info, exog)
        if len(params) > exog.shape[1]:
            params = params[0:exog.shape[1]]
        return np.dot(exog, params)