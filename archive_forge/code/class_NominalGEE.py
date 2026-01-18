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
class NominalGEE(GEE):
    __doc__ = '    Nominal Response Marginal Regression Model using GEE.\n' + _gee_init_doc % {'extra_params': base._missing_param_doc, 'family_doc': _gee_nominal_family_doc, 'example': _gee_nominal_example, 'notes': _gee_nointercept}

    def __init__(self, endog, exog, groups, time=None, family=None, cov_struct=None, missing='none', offset=None, dep_data=None, constraint=None, **kwargs):
        endog, exog, groups, time, offset = self.setup_nominal(endog, exog, groups, time, offset)
        if family is None:
            family = _Multinomial(self.ncut + 1)
        if cov_struct is None:
            cov_struct = cov_structs.NominalIndependence()
        super().__init__(endog, exog, groups, time, family, cov_struct, missing, offset, dep_data, constraint)

    def _starting_params(self):
        exposure = getattr(self, 'exposure', None)
        model = GEE(self.endog, self.exog, self.groups, time=self.time, family=families.Binomial(), offset=self.offset, exposure=exposure)
        result = model.fit()
        return result.params

    def setup_nominal(self, endog, exog, groups, time, offset):
        """
        Restructure nominal data as binary indicators so that they can
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
        self.ncut = ncut
        nrows = len(endog_cuts) * exog.shape[0]
        ncols = len(endog_cuts) * exog.shape[1]
        exog_out = np.zeros((nrows, ncols), dtype=np.float64)
        endog_out = np.zeros(nrows, dtype=np.float64)
        groups_out = np.zeros(nrows, dtype=np.float64)
        time_out = np.zeros((nrows, time.shape[1]), dtype=np.float64)
        offset_out = np.zeros(nrows, dtype=np.float64)
        jrow = 0
        zipper = zip(exog, endog, groups, time, offset)
        for exog_row, endog_value, group_value, time_value, offset_value in zipper:
            for thresh_ix, thresh in enumerate(endog_cuts):
                u = np.zeros(len(endog_cuts), dtype=np.float64)
                u[thresh_ix] = 1
                exog_out[jrow, :] = np.kron(u, exog_row)
                endog_out[jrow] = int(endog_value == thresh)
                groups_out[jrow] = group_value
                time_out[jrow] = time_value
                offset_out[jrow] = offset_value
                jrow += 1
        if isinstance(self.exog_orig, pd.DataFrame):
            xnames_in = self.exog_orig.columns
        else:
            xnames_in = ['x%d' % k for k in range(1, exog.shape[1] + 1)]
        xnames = []
        for tr in endog_cuts:
            xnames.extend(['{}[{:.1f}]'.format(v, tr) for v in xnames_in])
        exog_out = pd.DataFrame(exog_out, columns=xnames)
        exog_out = pd.DataFrame(exog_out, columns=xnames)
        if isinstance(self.endog_orig, pd.Series):
            endog_out = pd.Series(endog_out, name=self.endog_orig.name)
        return (endog_out, exog_out, groups_out, time_out, offset_out)

    def mean_deriv(self, exog, lin_pred):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        exog : array_like
           The exogeneous data at which the derivative is computed,
           number of rows must be a multiple of `ncut`.
        lin_pred : array_like
           The values of the linear predictor, length must be multiple
           of `ncut`.

        Returns
        -------
        The derivative of the expected endog with respect to the
        parameters.
        """
        expval = np.exp(lin_pred)
        expval_m = np.reshape(expval, (len(expval) // self.ncut, self.ncut))
        denom = 1 + expval_m.sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))
        mprob = expval / denom
        dmat = mprob[:, None] * exog
        ddenom = expval[:, None] * exog
        dmat -= mprob[:, None] * ddenom / denom[:, None]
        return dmat

    def mean_deriv_exog(self, exog, params, offset_exposure=None):
        """
        Derivative of the expected endog with respect to exog for the
        multinomial model, used in analyzing marginal effects.

        Parameters
        ----------
        exog : array_like
           The exogeneous data at which the derivative is computed,
           number of rows must be a multiple of `ncut`.
        lpr : array_like
           The linear predictor values, length must be multiple of
           `ncut`.

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to exog.

        Notes
        -----
        offset_exposure must be set at None for the multinomial family.
        """
        if offset_exposure is not None:
            warnings.warn('Offset/exposure ignored for the multinomial family', ValueWarning)
        lpr = np.dot(exog, params)
        expval = np.exp(lpr)
        expval_m = np.reshape(expval, (len(expval) // self.ncut, self.ncut))
        denom = 1 + expval_m.sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))
        bmat0 = np.outer(np.ones(exog.shape[0]), params)
        qmat = []
        for j in range(self.ncut):
            ee = np.zeros(self.ncut, dtype=np.float64)
            ee[j] = 1
            qmat.append(np.kron(ee, np.ones(len(params) // self.ncut)))
        qmat = np.array(qmat)
        qmat = np.kron(np.ones((exog.shape[0] // self.ncut, 1)), qmat)
        bmat = bmat0 * qmat
        dmat = expval[:, None] * bmat / denom[:, None]
        expval_mb = np.kron(expval_m, np.ones((self.ncut, 1)))
        expval_mb = np.kron(expval_mb, np.ones((1, self.ncut)))
        dmat -= expval[:, None] * (bmat * expval_mb) / denom[:, None] ** 2
        return dmat

    @Appender(_gee_fit_doc)
    def fit(self, maxiter=60, ctol=1e-06, start_params=None, params_niter=1, first_dep_update=0, cov_type='robust'):
        rslt = super().fit(maxiter, ctol, start_params, params_niter, first_dep_update, cov_type=cov_type)
        if rslt is None:
            warnings.warn('GEE updates did not converge', ConvergenceWarning)
            return None
        rslt = rslt._results
        res_kwds = {k: getattr(rslt, k) for k in rslt._props}
        nom_rslt = NominalGEEResults(self, rslt.params, rslt.cov_params() / rslt.scale, rslt.scale, cov_type=cov_type, attr_kwds=res_kwds)
        return NominalGEEResultsWrapper(nom_rslt)