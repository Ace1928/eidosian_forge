from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
class OrdinalGEE(GEE):
    __doc__ = '    Ordinal Response Marginal Regression Model using GEE\n' + _gee_init_doc % {'extra_params': base._missing_param_doc, 'family_doc': _gee_ordinal_family_doc, 'example': _gee_ordinal_example, 'notes': _gee_nointercept}

    def __init__(self, endog, exog, groups, time=None, family=None, cov_struct=None, missing='none', offset=None, dep_data=None, constraint=None, **kwargs):
        if family is None:
            family = families.Binomial()
        elif not isinstance(family, families.Binomial):
            raise ValueError('ordinal GEE must use a Binomial family')
        if cov_struct is None:
            cov_struct = cov_structs.OrdinalIndependence()
        endog, exog, groups, time, offset = self.setup_ordinal(endog, exog, groups, time, offset)
        super().__init__(endog, exog, groups, time, family, cov_struct, missing, offset, dep_data, constraint)

    def setup_ordinal(self, endog, exog, groups, time, offset):
        """
        Restructure ordinal data as binary indicators so that they can
        be analyzed using Generalized Estimating Equations.
        """
        self.endog_orig = endog.copy()
        self.exog_orig = exog.copy()
        self.groups_orig = groups.copy()
        if offset is not None:
            self.offset_orig = offset.copy()
        else:
            self.offset_orig = None
            offset = np.zeros(len(endog))
        if time is not None:
            self.time_orig = time.copy()
        else:
            self.time_orig = None
            time = np.zeros((len(endog), 1))
        exog = np.asarray(exog)
        endog = np.asarray(endog)
        groups = np.asarray(groups)
        time = np.asarray(time)
        offset = np.asarray(offset)
        self.endog_values = np.unique(endog)
        endog_cuts = self.endog_values[0:-1]
        ncut = len(endog_cuts)
        nrows = ncut * len(endog)
        exog_out = np.zeros((nrows, exog.shape[1]), dtype=np.float64)
        endog_out = np.zeros(nrows, dtype=np.float64)
        intercepts = np.zeros((nrows, ncut), dtype=np.float64)
        groups_out = np.zeros(nrows, dtype=groups.dtype)
        time_out = np.zeros((nrows, time.shape[1]), dtype=np.float64)
        offset_out = np.zeros(nrows, dtype=np.float64)
        jrow = 0
        zipper = zip(exog, endog, groups, time, offset)
        for exog_row, endog_value, group_value, time_value, offset_value in zipper:
            for thresh_ix, thresh in enumerate(endog_cuts):
                exog_out[jrow, :] = exog_row
                endog_out[jrow] = int(np.squeeze(endog_value > thresh))
                intercepts[jrow, thresh_ix] = 1
                groups_out[jrow] = group_value
                time_out[jrow] = time_value
                offset_out[jrow] = offset_value
                jrow += 1
        exog_out = np.concatenate((intercepts, exog_out), axis=1)
        xnames = ['I(y>%.1f)' % v for v in endog_cuts]
        if type(self.exog_orig) is pd.DataFrame:
            xnames.extend(self.exog_orig.columns)
        else:
            xnames.extend(['x%d' % k for k in range(1, exog.shape[1] + 1)])
        exog_out = pd.DataFrame(exog_out, columns=xnames)
        if type(self.endog_orig) is pd.Series:
            endog_out = pd.Series(endog_out, name=self.endog_orig.name)
        return (endog_out, exog_out, groups_out, time_out, offset_out)

    def _starting_params(self):
        exposure = getattr(self, 'exposure', None)
        model = GEE(self.endog, self.exog, self.groups, time=self.time, family=families.Binomial(), offset=self.offset, exposure=exposure)
        result = model.fit()
        return result.params

    @Appender(_gee_fit_doc)
    def fit(self, maxiter=60, ctol=1e-06, start_params=None, params_niter=1, first_dep_update=0, cov_type='robust'):
        rslt = super().fit(maxiter, ctol, start_params, params_niter, first_dep_update, cov_type=cov_type)
        rslt = rslt._results
        res_kwds = {k: getattr(rslt, k) for k in rslt._props}
        ord_rslt = OrdinalGEEResults(self, rslt.params, rslt.cov_params() / rslt.scale, rslt.scale, cov_type=cov_type, attr_kwds=res_kwds)
        return OrdinalGEEResultsWrapper(ord_rslt)