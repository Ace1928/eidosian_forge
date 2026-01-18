import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
class MICE:
    __doc__ = "    Multiple Imputation with Chained Equations.\n\n    This class can be used to fit most statsmodels models to data sets\n    with missing values using the 'multiple imputation with chained\n    equations' (MICE) approach..\n\n    Parameters\n    ----------\n    model_formula : str\n        The model formula to be fit to the imputed data sets.  This\n        formula is for the 'analysis model'.\n    model_class : statsmodels model\n        The model to be fit to the imputed data sets.  This model\n        class if for the 'analysis model'.\n    data : MICEData instance\n        MICEData object containing the data set for which\n        missing values will be imputed\n    n_skip : int\n        The number of imputed datasets to skip between consecutive\n        imputed datasets that are used for analysis.\n    init_kwds : dict-like\n        Dictionary of keyword arguments passed to the init method\n        of the analysis model.\n    fit_kwds : dict-like\n        Dictionary of keyword arguments passed to the fit method\n        of the analysis model.\n\n    Examples\n    --------\n    Run all MICE steps and obtain results:\n    {mice_example_1}\n\n    Obtain a sequence of fitted analysis models without combining\n    to obtain summary::\n    {mice_example_2}\n    ".format(mice_example_1=_mice_example_1, mice_example_2=_mice_example_2)

    def __init__(self, model_formula, model_class, data, n_skip=3, init_kwds=None, fit_kwds=None):
        self.model_formula = model_formula
        self.model_class = model_class
        self.n_skip = n_skip
        self.data = data
        self.results_list = []
        self.init_kwds = init_kwds if init_kwds is not None else {}
        self.fit_kwds = fit_kwds if fit_kwds is not None else {}

    def next_sample(self):
        """
        Perform one complete MICE iteration.

        A single MICE iteration updates all missing values using their
        respective imputation models, then fits the analysis model to
        the imputed data.

        Returns
        -------
        params : array_like
            The model parameters for the analysis model.

        Notes
        -----
        This function fits the analysis model and returns its
        parameter estimate.  The parameter vector is not stored by the
        class and is not used in any subsequent calls to `combine`.
        Use `fit` to run all MICE steps together and obtain summary
        results.

        The complete cycle of missing value imputation followed by
        fitting the analysis model is repeated `n_skip + 1` times and
        the analysis model parameters from the final fit are returned.
        """
        self.data.update_all(self.n_skip + 1)
        start_params = None
        if len(self.results_list) > 0:
            start_params = self.results_list[-1].params
        model = self.model_class.from_formula(self.model_formula, self.data.data, **self.init_kwds)
        self.fit_kwds.update({'start_params': start_params})
        result = model.fit(**self.fit_kwds)
        return result

    def fit(self, n_burnin=10, n_imputations=10):
        """
        Fit a model using MICE.

        Parameters
        ----------
        n_burnin : int
            The number of burn-in cycles to skip.
        n_imputations : int
            The number of data sets to impute
        """
        self.data.update_all(n_burnin)
        for j in range(n_imputations):
            result = self.next_sample()
            self.results_list.append(result)
        self.endog_names = result.model.endog_names
        self.exog_names = result.model.exog_names
        return self.combine()

    def combine(self):
        """
        Pools MICE imputation results.

        This method can only be used after the `run` method has been
        called.  Returns estimates and standard errors of the analysis
        model parameters.

        Returns a MICEResults instance.
        """
        params_list = []
        cov_within = 0.0
        scale_list = []
        for results in self.results_list:
            results_uw = results._results
            params_list.append(results_uw.params)
            cov_within += results_uw.cov_params()
            scale_list.append(results.scale)
        params_list = np.asarray(params_list)
        scale_list = np.asarray(scale_list)
        params = params_list.mean(0)
        cov_within /= len(self.results_list)
        cov_between = np.cov(params_list.T)
        f = 1 + 1 / float(len(self.results_list))
        cov_params = cov_within + f * cov_between
        fmi = f * np.diag(cov_between) / np.diag(cov_params)
        scale = np.mean(scale_list)
        results = MICEResults(self, params, cov_params / scale)
        results.scale = scale
        results.frac_miss_info = fmi
        results.exog_names = self.exog_names
        results.endog_names = self.endog_names
        results.model_class = self.model_class
        return results