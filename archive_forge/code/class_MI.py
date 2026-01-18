import numpy as np
import pandas as pd
from statsmodels.base.model import LikelihoodModelResults
class MI:
    """
    MI performs multiple imputation using a provided imputer object.

    Parameters
    ----------
    imp : object
        An imputer class, such as BayesGaussMI.
    model : model class
        Any statsmodels model class.
    model_args_fn : function
        A function taking an imputed dataset as input and returning
        endog, exog.  If the model is fit using a formula, returns
        a DataFrame used to build the model.  Optional when a formula
        is used.
    model_kwds_fn : function, optional
        A function taking an imputed dataset as input and returning
        a dictionary of model keyword arguments.
    formula : str, optional
        If provided, the model is constructed using the `from_formula`
        class method, otherwise the `__init__` method is used.
    fit_args : list-like, optional
        List of arguments to be passed to the fit method
    fit_kwds : dict-like, optional
        Keyword arguments to be passed to the fit method
    xfunc : function mapping ndarray to ndarray
        A function that is applied to the complete data matrix
        prior to fitting the model
    burn : int
        Number of burn-in iterations
    nrep : int
        Number of imputed data sets to use in the analysis
    skip : int
        Number of Gibbs iterations to skip between successive
        multiple imputation fits.

    Notes
    -----
    The imputer object must have an 'update' method, and a 'data'
    attribute that contains the current imputed dataset.

    xfunc can be used to introduce domain constraints, e.g. when
    imputing binary data the imputed continuous values can be rounded
    to 0/1.
    """

    def __init__(self, imp, model, model_args_fn=None, model_kwds_fn=None, formula=None, fit_args=None, fit_kwds=None, xfunc=None, burn=100, nrep=20, skip=10):
        self.imp = imp
        self.skip = skip
        self.model = model
        self.formula = formula
        if model_args_fn is None:

            def f(x):
                return []
            model_args_fn = f
        self.model_args_fn = model_args_fn
        if model_kwds_fn is None:

            def f(x):
                return {}
            model_kwds_fn = f
        self.model_kwds_fn = model_kwds_fn
        if fit_args is None:

            def f(x):
                return []
            fit_args = f
        self.fit_args = fit_args
        if fit_kwds is None:

            def f(x):
                return {}
            fit_kwds = f
        self.fit_kwds = fit_kwds
        self.xfunc = xfunc
        self.nrep = nrep
        self.skip = skip
        for k in range(burn):
            imp.update()

    def fit(self, results_cb=None):
        """
        Impute datasets, fit models, and pool results.

        Parameters
        ----------
        results_cb : function, optional
            If provided, each results instance r is passed through `results_cb`,
            then appended to the `results` attribute of the MIResults object.
            To save complete results, use `results_cb=lambda x: x`.  The default
            behavior is to save no results.

        Returns
        -------
        A MIResults object.
        """
        par, cov = ([], [])
        all_results = []
        for k in range(self.nrep):
            for k in range(self.skip + 1):
                self.imp.update()
            da = self.imp.data
            if self.xfunc is not None:
                da = self.xfunc(da)
            if self.formula is None:
                model = self.model(*self.model_args_fn(da), **self.model_kwds_fn(da))
            else:
                model = self.model.from_formula(self.formula, *self.model_args_fn(da), **self.model_kwds_fn(da))
            result = model.fit(*self.fit_args(da), **self.fit_kwds(da))
            if results_cb is not None:
                all_results.append(results_cb(result))
            par.append(np.asarray(result.params.copy()))
            cov.append(np.asarray(result.cov_params().copy()))
        params, cov_params, fmi = self._combine(par, cov)
        r = MIResults(self, model, params, cov_params)
        r.fmi = fmi
        r.results = all_results
        return r

    def _combine(self, par, cov):
        par = np.asarray(par)
        m = par.shape[0]
        params = par.mean(0)
        wcov = sum(cov) / len(cov)
        bcov = np.cov(par.T)
        bcov = np.atleast_2d(bcov)
        covp = wcov + (1 + 1 / float(m)) * bcov
        fmi = (1 + 1 / float(m)) * np.diag(bcov) / np.diag(covp)
        return (params, covp, fmi)