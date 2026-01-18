import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
class MICEResults(LikelihoodModelResults):

    def __init__(self, model, params, normalized_cov_params):
        super().__init__(model, params, normalized_cov_params)

    def summary(self, title=None, alpha=0.05):
        """
        Summarize the results of running MICE.

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
        info['Method:'] = 'MICE'
        info['Model:'] = self.model_class.__name__
        info['Dependent variable:'] = self.endog_names
        info['Sample size:'] = '%d' % self.model.data.data.shape[0]
        info['Scale'] = '%.2f' % self.scale
        info['Num. imputations'] = '%d' % len(self.model.results_list)
        smry.add_dict(info, align='l', float_format=float_format)
        param = summary2.summary_params(self, alpha=alpha)
        param['FMI'] = self.frac_miss_info
        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)
        return smry