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
class MultinomialResults(DiscreteResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for multinomial data', 'extra_attr': ''}

    def __init__(self, model, mlefit):
        super().__init__(model, mlefit)
        self.J = model.J
        self.K = model.K

    @staticmethod
    def _maybe_convert_ynames_int(ynames):
        issue_warning = False
        msg = 'endog contains values are that not int-like. Uses string representation of value. Use integer-valued endog to suppress this warning.'
        for i in ynames:
            try:
                if ynames[i] % 1 == 0:
                    ynames[i] = str(int(ynames[i]))
                else:
                    issue_warning = True
                    ynames[i] = str(ynames[i])
            except TypeError:
                ynames[i] = str(ynames[i])
        if issue_warning:
            warnings.warn(msg, SpecificationWarning)
        return ynames

    def _get_endog_name(self, yname, yname_list, all=False):
        """
        If all is False, the first variable name is dropped
        """
        model = self.model
        if yname is None:
            yname = model.endog_names
        if yname_list is None:
            ynames = model._ynames_map
            ynames = self._maybe_convert_ynames_int(ynames)
            ynames = [ynames[key] for key in range(int(model.J))]
            ynames = ['='.join([yname, name]) for name in ynames]
            if not all:
                yname_list = ynames[1:]
            else:
                yname_list = ynames
        return (yname, yname_list)

    def pred_table(self):
        """
        Returns the J x J prediction table.

        Notes
        -----
        pred_table[i,j] refers to the number of times "i" was observed and
        the model predicted "j". Correct predictions are along the diagonal.
        """
        ju = self.model.J - 1
        bins = np.concatenate(([0], np.linspace(0.5, ju - 0.5, ju), [ju]))
        return np.histogram2d(self.model.endog, self.predict().argmax(1), bins=bins)[0]

    @cache_readonly
    def bse(self):
        bse = np.sqrt(np.diag(self.cov_params()))
        return bse.reshape(self.params.shape, order='F')

    @cache_readonly
    def aic(self):
        return -2 * (self.llf - (self.df_model + self.model.J - 1))

    @cache_readonly
    def bic(self):
        return -2 * self.llf + np.log(self.nobs) * (self.df_model + self.model.J - 1)

    def conf_int(self, alpha=0.05, cols=None):
        confint = super(DiscreteResults, self).conf_int(alpha=alpha, cols=cols)
        return confint.transpose(2, 0, 1)

    def get_prediction(self):
        """Not implemented for Multinomial
        """
        raise NotImplementedError

    def margeff(self):
        raise NotImplementedError('Use get_margeff instead')

    @cache_readonly
    def resid_misclassified(self):
        """
        Residuals indicating which observations are misclassified.

        Notes
        -----
        The residuals for the multinomial model are defined as

        .. math:: argmax(y_i) \\neq argmax(p_i)

        where :math:`argmax(y_i)` is the index of the category for the
        endogenous variable and :math:`argmax(p_i)` is the index of the
        predicted probabilities for each category. That is, the residual
        is a binary indicator that is 0 if the category with the highest
        predicted probability is the same as that of the observed variable
        and 1 otherwise.
        """
        return (self.model.wendog.argmax(1) != self.predict().argmax(1)).astype(float)

    def summary2(self, alpha=0.05, float_format='%.4f'):
        """Experimental function to summarize regression results

        Parameters
        ----------
        alpha : float
            significance level for the confidence intervals
        float_format : str
            print format for floats in parameters summary

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_dict(summary2.summary_model(self))
        eqn = self.params.shape[1]
        confint = self.conf_int(alpha)
        for i in range(eqn):
            coefs = summary2.summary_params((self, self.params[:, i], self.bse[:, i], self.tvalues[:, i], self.pvalues[:, i], confint[i]), alpha=alpha)
            level_str = self.model.endog_names + ' = ' + str(i)
            coefs[level_str] = coefs.index
            coefs = coefs.iloc[:, [-1, 0, 1, 2, 3, 4, 5]]
            smry.add_df(coefs, index=False, header=True, float_format=float_format)
            smry.add_title(results=self)
        return smry