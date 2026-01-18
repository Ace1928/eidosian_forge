import numpy as np
import pandas as pd
from statsmodels.base.model import LikelihoodModelResults
class MIResults(LikelihoodModelResults):
    """
    A results class for multiple imputation (MI).

    Parameters
    ----------
    mi : MI instance
        The MI object that produced the results
    model : instance of statsmodels model class
        This can be any instance from the multiple imputation runs.
        It is used to get class information, the specific parameter
        and data values are not used.
    params : array_like
        The overall multiple imputation parameter estimates.
    normalized_cov_params : array_like (2d)
        The overall variance covariance matrix of the estimates.
    """

    def __init__(self, mi, model, params, normalized_cov_params):
        super().__init__(model, params, normalized_cov_params)
        self.mi = mi
        self._model = model

    def summary(self, title=None, alpha=0.05):
        """
        Summarize the results of running multiple imputation.

        Parameters
        ----------
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            Significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be
            printed or converted to various output formats.
        """
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        float_format = '%8.3f'
        info = {}
        info['Method:'] = 'MI'
        info['Model:'] = self.mi.model.__name__
        info['Dependent variable:'] = self._model.endog_names
        info['Sample size:'] = '%d' % self.mi.imp.data.shape[0]
        info['Num. imputations'] = '%d' % self.mi.nrep
        smry.add_dict(info, align='l', float_format=float_format)
        param = summary2.summary_params(self, alpha=alpha)
        param['FMI'] = self.fmi
        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)
        return smry