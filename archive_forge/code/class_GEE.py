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
class GEE(GLM):
    __doc__ = '    Marginal Regression Model using Generalized Estimating Equations.\n' + _gee_init_doc % {'extra_params': base._missing_param_doc, 'family_doc': _gee_family_doc, 'example': _gee_example, 'notes': ''}
    cached_means = None

    def __init__(self, endog, exog, groups, time=None, family=None, cov_struct=None, missing='none', offset=None, exposure=None, dep_data=None, constraint=None, update_dep=True, weights=None, **kwargs):
        if type(self) is GEE:
            self._check_kwargs(kwargs)
        if family is not None:
            if not isinstance(family.link, tuple(family.safe_links)):
                msg = 'The {0} link function does not respect the domain of the {1} family.'
                warnings.warn(msg.format(family.link.__class__.__name__, family.__class__.__name__), DomainWarning)
        groups = np.asarray(groups)
        if 'missing_idx' in kwargs and kwargs['missing_idx'] is not None:
            ii = ~kwargs['missing_idx']
            groups = groups[ii]
            if time is not None:
                time = time[ii]
            if offset is not None:
                offset = offset[ii]
            if exposure is not None:
                exposure = exposure[ii]
            del kwargs['missing_idx']
        self.missing = missing
        self.dep_data = dep_data
        self.constraint = constraint
        self.update_dep = update_dep
        self._fit_history = defaultdict(list)
        super().__init__(endog, exog, groups=groups, time=time, offset=offset, exposure=exposure, weights=weights, dep_data=dep_data, missing=missing, family=family, **kwargs)
        _check_args(self.endog, self.exog, self.groups, self.time, getattr(self, 'offset', None), getattr(self, 'exposure', None))
        self._init_keys.extend(['update_dep', 'constraint', 'family', 'cov_struct'])
        try:
            self._init_keys.remove('freq_weights')
            self._init_keys.remove('var_weights')
        except ValueError:
            pass
        if family is None:
            family = families.Gaussian()
        elif not issubclass(family.__class__, families.Family):
            raise ValueError('GEE: `family` must be a genmod family instance')
        self.family = family
        if cov_struct is None:
            cov_struct = cov_structs.Independence()
        elif not issubclass(cov_struct.__class__, cov_structs.CovStruct):
            raise ValueError('GEE: `cov_struct` must be a genmod cov_struct instance')
        self.cov_struct = cov_struct
        self.constraint = None
        if constraint is not None:
            if len(constraint) != 2:
                raise ValueError('GEE: `constraint` must be a 2-tuple.')
            if constraint[0].shape[1] != self.exog.shape[1]:
                raise ValueError('GEE: the left hand side of the constraint must have the same number of columns as the exog matrix.')
            self.constraint = ParameterConstraint(constraint[0], constraint[1], self.exog)
            if self._offset_exposure is not None:
                self._offset_exposure += self.constraint.offset_increment()
            else:
                self._offset_exposure = self.constraint.offset_increment().copy()
            self.exog = self.constraint.reduced_exog()
        group_labels, ix = np.unique(self.groups, return_inverse=True)
        se = pd.Series(index=np.arange(len(ix)), dtype='int')
        gb = se.groupby(ix).groups
        dk = [(lb, np.asarray(gb[k])) for k, lb in enumerate(group_labels)]
        self.group_indices = dict(dk)
        self.group_labels = group_labels
        self.endog_li = self.cluster_list(self.endog)
        self.exog_li = self.cluster_list(self.exog)
        if self.weights is not None:
            self.weights_li = self.cluster_list(self.weights)
        self.num_group = len(self.endog_li)
        if self.time is not None:
            if self.time.ndim == 1:
                self.time = self.time[:, None]
            self.time_li = self.cluster_list(self.time)
        else:
            self.time_li = [np.arange(len(y), dtype=np.float64)[:, None] for y in self.endog_li]
            self.time = np.concatenate(self.time_li)
        if self._offset_exposure is None or (np.isscalar(self._offset_exposure) and self._offset_exposure == 0.0):
            self.offset_li = None
        else:
            self.offset_li = self.cluster_list(self._offset_exposure)
        if constraint is not None:
            self.constraint.exog_fulltrans_li = self.cluster_list(self.constraint.exog_fulltrans)
        self.family = family
        self.cov_struct.initialize(self)
        group_ns = [len(y) for y in self.endog_li]
        self.nobs = sum(group_ns)
        self.df_model = self.exog.shape[1] - 1
        self.df_resid = self.nobs - self.exog.shape[1]
        maxgroup = max([len(x) for x in self.endog_li])
        if maxgroup == 1:
            self.update_dep = False

    @classmethod
    def from_formula(cls, formula, groups, data, subset=None, time=None, offset=None, exposure=None, *args, **kwargs):
        '\n        Create a GEE model instance from a formula and dataframe.\n\n        Parameters\n        ----------\n        formula : str or generic Formula object\n            The formula specifying the model\n        groups : array_like or string\n            Array of grouping labels.  If a string, this is the name\n            of a variable in `data` that contains the grouping labels.\n        data : array_like\n            The data for the model.\n        subset : array_like\n            An array-like object of booleans, integers, or index\n            values that indicate the subset of the data to used when\n            fitting the model.\n        time : array_like or string\n            The time values, used for dependence structures involving\n            distances between observations.  If a string, this is the\n            name of a variable in `data` that contains the time\n            values.\n        offset : array_like or string\n            The offset values, added to the linear predictor.  If a\n            string, this is the name of a variable in `data` that\n            contains the offset values.\n        exposure : array_like or string\n            The exposure values, only used if the link function is the\n            logarithm function, in which case the log of `exposure`\n            is added to the offset (if any).  If a string, this is the\n            name of a variable in `data` that contains the offset\n            values.\n        {missing_param_doc}\n        args : extra arguments\n            These are passed to the model\n        kwargs : extra keyword arguments\n            These are passed to the model with two exceptions. `dep_data`\n            is processed as described below.  The ``eval_env`` keyword is\n            passed to patsy. It can be either a\n            :class:`patsy:patsy.EvalEnvironment` object or an integer\n            indicating the depth of the namespace to use. For example, the\n            default ``eval_env=0`` uses the calling namespace.\n            If you wish to use a "clean" environment set ``eval_env=-1``.\n\n        Optional arguments\n        ------------------\n        dep_data : str or array_like\n            Data used for estimating the dependence structure.  See\n            specific dependence structure classes (e.g. Nested) for\n            details.  If `dep_data` is a string, it is interpreted as\n            a formula that is applied to `data`. If it is an array, it\n            must be an array of strings corresponding to column names in\n            `data`.  Otherwise it must be an array-like with the same\n            number of rows as data.\n\n        Returns\n        -------\n        model : GEE model instance\n\n        Notes\n        -----\n        `data` must define __getitem__ with the keys in the formula\n        terms args and kwargs are passed on to the model\n        instantiation. E.g., a numpy structured or rec array, a\n        dictionary, or a pandas DataFrame.\n        '.format(missing_param_doc=base._missing_param_doc)
        groups_name = 'Groups'
        if isinstance(groups, str):
            groups_name = groups
            groups = data[groups]
        if isinstance(time, str):
            time = data[time]
        if isinstance(offset, str):
            offset = data[offset]
        if isinstance(exposure, str):
            exposure = data[exposure]
        dep_data = kwargs.get('dep_data')
        dep_data_names = None
        if dep_data is not None:
            if isinstance(dep_data, str):
                dep_data = patsy.dmatrix(dep_data, data, return_type='dataframe')
                dep_data_names = dep_data.columns.tolist()
            else:
                dep_data_names = list(dep_data)
                dep_data = data[dep_data]
            kwargs['dep_data'] = np.asarray(dep_data)
        family = None
        if 'family' in kwargs:
            family = kwargs['family']
            del kwargs['family']
        model = super().from_formula(formula, *args, data=data, subset=subset, groups=groups, time=time, offset=offset, exposure=exposure, family=family, **kwargs)
        if dep_data_names is not None:
            model._dep_data_names = dep_data_names
        model._groups_name = groups_name
        return model

    def cluster_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        cluster structure.
        """
        if array.ndim == 1:
            return [np.array(array[self.group_indices[k]]) for k in self.group_labels]
        else:
            return [np.array(array[self.group_indices[k], :]) for k in self.group_labels]

    def compare_score_test(self, submodel):
        """
        Perform a score test for the given submodel against this model.

        Parameters
        ----------
        submodel : GEEResults instance
            A fitted GEE model that is a submodel of this model.

        Returns
        -------
        A dictionary with keys "statistic", "p-value", and "df",
        containing the score test statistic, its chi^2 p-value,
        and the degrees of freedom used to compute the p-value.

        Notes
        -----
        The score test can be performed without calling 'fit' on the
        larger model.  The provided submodel must be obtained from a
        fitted GEE.

        This method performs the same score test as can be obtained by
        fitting the GEE with a linear constraint and calling `score_test`
        on the results.

        References
        ----------
        Xu Guo and Wei Pan (2002). "Small sample performance of the score
        test in GEE".
        http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf
        """
        self.scaletype = submodel.model.scaletype
        submod = submodel.model
        if self.exog.shape[0] != submod.exog.shape[0]:
            msg = 'Model and submodel have different numbers of cases.'
            raise ValueError(msg)
        if self.exog.shape[1] == submod.exog.shape[1]:
            msg = 'Model and submodel have the same number of variables'
            warnings.warn(msg)
        if not isinstance(self.family, type(submod.family)):
            msg = 'Model and submodel have different GLM families.'
            warnings.warn(msg)
        if not isinstance(self.cov_struct, type(submod.cov_struct)):
            warnings.warn('Model and submodel have different GEE covariance structures.')
        if not np.equal(self.weights, submod.weights).all():
            msg = 'Model and submodel should have the same weights.'
            warnings.warn(msg)
        qm, qc = _score_test_submodel(self, submodel.model)
        if qm is None:
            msg = 'The provided model is not a submodel.'
            raise ValueError(msg)
        params_ex = np.dot(qm, submodel.params)
        cov_struct_save = self.cov_struct
        import copy
        cached_means_save = copy.deepcopy(self.cached_means)
        self.cov_struct = submodel.cov_struct
        self.update_cached_means(params_ex)
        _, score = self._update_mean_params()
        if score is None:
            msg = 'Singular matrix encountered in GEE score test'
            warnings.warn(msg, ConvergenceWarning)
            return None
        if not hasattr(self, 'ddof_scale'):
            self.ddof_scale = self.exog.shape[1]
        if not hasattr(self, 'scaling_factor'):
            self.scaling_factor = 1
        _, ncov1, cmat = self._covmat()
        score2 = np.dot(qc.T, score)
        try:
            amat = np.linalg.inv(ncov1)
        except np.linalg.LinAlgError:
            amat = np.linalg.pinv(ncov1)
        bmat_11 = np.dot(qm.T, np.dot(cmat, qm))
        bmat_22 = np.dot(qc.T, np.dot(cmat, qc))
        bmat_12 = np.dot(qm.T, np.dot(cmat, qc))
        amat_11 = np.dot(qm.T, np.dot(amat, qm))
        amat_12 = np.dot(qm.T, np.dot(amat, qc))
        try:
            ab = np.linalg.solve(amat_11, bmat_12)
        except np.linalg.LinAlgError:
            ab = np.dot(np.linalg.pinv(amat_11), bmat_12)
        score_cov = bmat_22 - np.dot(amat_12.T, ab)
        try:
            aa = np.linalg.solve(amat_11, amat_12)
        except np.linalg.LinAlgError:
            aa = np.dot(np.linalg.pinv(amat_11), amat_12)
        score_cov -= np.dot(bmat_12.T, aa)
        try:
            ab = np.linalg.solve(amat_11, bmat_11)
        except np.linalg.LinAlgError:
            ab = np.dot(np.linalg.pinv(amat_11), bmat_11)
        try:
            aa = np.linalg.solve(amat_11, amat_12)
        except np.linalg.LinAlgError:
            aa = np.dot(np.linalg.pinv(amat_11), amat_12)
        score_cov += np.dot(amat_12.T, np.dot(ab, aa))
        self.cov_struct = cov_struct_save
        self.cached_means = cached_means_save
        from scipy.stats.distributions import chi2
        try:
            sc2 = np.linalg.solve(score_cov, score2)
        except np.linalg.LinAlgError:
            sc2 = np.dot(np.linalg.pinv(score_cov), score2)
        score_statistic = np.dot(score2, sc2)
        score_df = len(score2)
        score_pvalue = 1 - chi2.cdf(score_statistic, score_df)
        return {'statistic': score_statistic, 'df': score_df, 'p-value': score_pvalue}

    def estimate_scale(self):
        """
        Estimate the dispersion/scale.
        """
        if self.scaletype is None:
            if isinstance(self.family, (families.Binomial, families.Poisson, families.NegativeBinomial, _Multinomial)):
                return 1.0
        elif isinstance(self.scaletype, float):
            return np.array(self.scaletype)
        endog = self.endog_li
        cached_means = self.cached_means
        nobs = self.nobs
        varfunc = self.family.variance
        scale = 0.0
        fsum = 0.0
        for i in range(self.num_group):
            if len(endog[i]) == 0:
                continue
            expval, _ = cached_means[i]
            sdev = np.sqrt(varfunc(expval))
            resid = (endog[i] - expval) / sdev
            if self.weights is not None:
                f = self.weights_li[i]
                scale += np.sum(f * resid ** 2)
                fsum += f.sum()
            else:
                scale += np.sum(resid ** 2)
                fsum += len(resid)
        scale /= fsum * (nobs - self.ddof_scale) / float(nobs)
        return scale

    def mean_deriv(self, exog, lin_pred):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        exog : array_like
           The exogeneous data at which the derivative is computed.
        lin_pred : array_like
           The values of the linear predictor.

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.

        Notes
        -----
        If there is an offset or exposure, it should be added to
        `lin_pred` prior to calling this function.
        """
        idl = self.family.link.inverse_deriv(lin_pred)
        dmat = exog * idl[:, None]
        return dmat

    def mean_deriv_exog(self, exog, params, offset_exposure=None):
        """
        Derivative of the expected endog with respect to exog.

        Parameters
        ----------
        exog : array_like
            Values of the independent variables at which the derivative
            is calculated.
        params : array_like
            Parameter values at which the derivative is calculated.
        offset_exposure : array_like, optional
            Combined offset and exposure.

        Returns
        -------
        The derivative of the expected endog with respect to exog.
        """
        lin_pred = np.dot(exog, params)
        if offset_exposure is not None:
            lin_pred += offset_exposure
        idl = self.family.link.inverse_deriv(lin_pred)
        dmat = np.outer(idl, params)
        return dmat

    def _update_mean_params(self):
        """
        Returns
        -------
        update : array_like
            The update vector such that params + update is the next
            iterate when solving the score equations.
        score : array_like
            The current value of the score equations, not
            incorporating the scale parameter.  If desired,
            multiply this vector by the scale parameter to
            incorporate the scale.
        """
        endog = self.endog_li
        exog = self.exog_li
        weights = getattr(self, 'weights_li', None)
        cached_means = self.cached_means
        varfunc = self.family.variance
        bmat, score = (0, 0)
        for i in range(self.num_group):
            expval, lpr = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))
            if weights is not None:
                w = weights[i]
                wresid = resid * w
                wdmat = dmat * w[:, None]
            else:
                wresid = resid
                wdmat = dmat
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (wdmat, wresid))
            if rslt is None:
                return (None, None)
            vinv_d, vinv_resid = tuple(rslt)
            bmat += np.dot(dmat.T, vinv_d)
            score += np.dot(dmat.T, vinv_resid)
        try:
            update = np.linalg.solve(bmat, score)
        except np.linalg.LinAlgError:
            update = np.dot(np.linalg.pinv(bmat), score)
        self._fit_history['cov_adjust'].append(self.cov_struct.cov_adjust)
        return (update, score)

    def update_cached_means(self, mean_params):
        """
        cached_means should always contain the most recent calculation
        of the group-wise mean vectors.  This function should be
        called every time the regression parameters are changed, to
        keep the cached means up to date.
        """
        endog = self.endog_li
        exog = self.exog_li
        offset = self.offset_li
        linkinv = self.family.link.inverse
        self.cached_means = []
        for i in range(self.num_group):
            if len(endog[i]) == 0:
                continue
            lpr = np.dot(exog[i], mean_params)
            if offset is not None:
                lpr += offset[i]
            expval = linkinv(lpr)
            self.cached_means.append((expval, lpr))

    def _covmat(self):
        """
        Returns the sampling covariance matrix of the regression
        parameters and related quantities.

        Returns
        -------
        cov_robust : array_like
           The robust, or sandwich estimate of the covariance, which
           is meaningful even if the working covariance structure is
           incorrectly specified.
        cov_naive : array_like
           The model-based estimate of the covariance, which is
           meaningful if the covariance structure is correctly
           specified.
        cmat : array_like
           The center matrix of the sandwich expression, used in
           obtaining score test results.
        """
        endog = self.endog_li
        exog = self.exog_li
        weights = getattr(self, 'weights_li', None)
        varfunc = self.family.variance
        cached_means = self.cached_means
        bmat, cmat = (0, 0)
        for i in range(self.num_group):
            expval, lpr = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))
            if weights is not None:
                w = weights[i]
                wresid = resid * w
                wdmat = dmat * w[:, None]
            else:
                wresid = resid
                wdmat = dmat
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (wdmat, wresid))
            if rslt is None:
                return (None, None, None, None)
            vinv_d, vinv_resid = tuple(rslt)
            bmat += np.dot(dmat.T, vinv_d)
            dvinv_resid = np.dot(dmat.T, vinv_resid)
            cmat += np.outer(dvinv_resid, dvinv_resid)
        scale = self.estimate_scale()
        try:
            bmati = np.linalg.inv(bmat)
        except np.linalg.LinAlgError:
            bmati = np.linalg.pinv(bmat)
        cov_naive = bmati * scale
        cov_robust = np.dot(bmati, np.dot(cmat, bmati))
        cov_naive *= self.scaling_factor
        cov_robust *= self.scaling_factor
        return (cov_robust, cov_naive, cmat)

    def _bc_covmat(self, cov_naive):
        cov_naive = cov_naive / self.scaling_factor
        endog = self.endog_li
        exog = self.exog_li
        varfunc = self.family.variance
        cached_means = self.cached_means
        scale = self.estimate_scale()
        bcm = 0
        for i in range(self.num_group):
            expval, lpr = cached_means[i]
            resid = endog[i] - expval
            dmat = self.mean_deriv(exog[i], lpr)
            sdev = np.sqrt(varfunc(expval))
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (dmat,))
            if rslt is None:
                return None
            vinv_d = rslt[0]
            vinv_d /= scale
            hmat = np.dot(vinv_d, cov_naive)
            hmat = np.dot(hmat, dmat.T).T
            f = self.weights_li[i] if self.weights is not None else 1.0
            aresid = np.linalg.solve(np.eye(len(resid)) - hmat, resid)
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (aresid,))
            if rslt is None:
                return None
            srt = rslt[0]
            srt = f * np.dot(dmat.T, srt) / scale
            bcm += np.outer(srt, srt)
        cov_robust_bc = np.dot(cov_naive, np.dot(bcm, cov_naive))
        cov_robust_bc *= self.scaling_factor
        return cov_robust_bc

    def _starting_params(self):
        if np.isscalar(self._offset_exposure):
            offset = None
        else:
            offset = self._offset_exposure
        model = GLM(self.endog, self.exog, family=self.family, offset=offset, freq_weights=self.weights)
        result = model.fit()
        return result.params

    @Appender(_gee_fit_doc)
    def fit(self, maxiter=60, ctol=1e-06, start_params=None, params_niter=1, first_dep_update=0, cov_type='robust', ddof_scale=None, scaling_factor=1.0, scale=None):
        self.scaletype = scale
        if ddof_scale is None:
            self.ddof_scale = self.exog.shape[1]
        else:
            if not ddof_scale >= 0:
                raise ValueError('ddof_scale must be a non-negative number or None')
            self.ddof_scale = ddof_scale
        self.scaling_factor = scaling_factor
        self._fit_history = defaultdict(list)
        if self.weights is not None and cov_type == 'naive':
            raise ValueError('when using weights, cov_type may not be naive')
        if start_params is None:
            mean_params = self._starting_params()
        else:
            start_params = np.asarray(start_params)
            mean_params = start_params.copy()
        self.update_cached_means(mean_params)
        del_params = -1.0
        num_assoc_updates = 0
        for itr in range(maxiter):
            update, score = self._update_mean_params()
            if update is None:
                warnings.warn('Singular matrix encountered in GEE update', ConvergenceWarning)
                break
            mean_params += update
            self.update_cached_means(mean_params)
            del_params = np.sqrt(np.sum(score ** 2))
            self._fit_history['params'].append(mean_params.copy())
            self._fit_history['score'].append(score)
            self._fit_history['dep_params'].append(self.cov_struct.dep_params)
            if del_params < ctol and (num_assoc_updates > 0 or self.update_dep is False):
                break
            if self.update_dep and itr % params_niter == 0 and (itr >= first_dep_update):
                self._update_assoc(mean_params)
                num_assoc_updates += 1
        if del_params >= ctol:
            warnings.warn('Iteration limit reached prior to convergence', IterationLimitWarning)
        if mean_params is None:
            warnings.warn('Unable to estimate GEE parameters.', ConvergenceWarning)
            return None
        bcov, ncov, _ = self._covmat()
        if bcov is None:
            warnings.warn('Estimated covariance structure for GEE estimates is singular', ConvergenceWarning)
            return None
        bc_cov = None
        if cov_type == 'bias_reduced':
            bc_cov = self._bc_covmat(ncov)
        if self.constraint is not None:
            x = mean_params.copy()
            mean_params, bcov = self._handle_constraint(mean_params, bcov)
            if mean_params is None:
                warnings.warn('Unable to estimate constrained GEE parameters.', ConvergenceWarning)
                return None
            y, ncov = self._handle_constraint(x, ncov)
            if y is None:
                warnings.warn('Unable to estimate constrained GEE parameters.', ConvergenceWarning)
                return None
            if bc_cov is not None:
                y, bc_cov = self._handle_constraint(x, bc_cov)
                if x is None:
                    warnings.warn('Unable to estimate constrained GEE parameters.', ConvergenceWarning)
                    return None
        scale = self.estimate_scale()
        res_kwds = dict(cov_type=cov_type, cov_robust=bcov, cov_naive=ncov, cov_robust_bc=bc_cov)
        results = GEEResults(self, mean_params, bcov / scale, scale, cov_type=cov_type, use_t=False, attr_kwds=res_kwds)
        results.fit_history = self._fit_history
        self.fit_history = defaultdict(list)
        results.score_norm = del_params
        results.converged = del_params < ctol
        results.cov_struct = self.cov_struct
        results.params_niter = params_niter
        results.first_dep_update = first_dep_update
        results.ctol = ctol
        results.maxiter = maxiter
        results._props = ['cov_type', 'use_t', 'cov_params_default', 'cov_robust', 'cov_naive', 'cov_robust_bc', 'fit_history', 'score_norm', 'converged', 'cov_struct', 'params_niter', 'first_dep_update', 'ctol', 'maxiter']
        return GEEResultsWrapper(results)

    def _update_regularized(self, params, pen_wt, scad_param, eps):
        sn, hm = (0, 0)
        for i in range(self.num_group):
            expval, _ = self.cached_means[i]
            resid = self.endog_li[i] - expval
            sdev = np.sqrt(self.family.variance(expval))
            ex = self.exog_li[i] * sdev[:, None] ** 2
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (resid, ex))
            sn0 = rslt[0]
            sn += np.dot(ex.T, sn0)
            hm0 = rslt[1]
            hm += np.dot(ex.T, hm0)
        ap = np.abs(params)
        clipped = np.clip(scad_param * pen_wt - ap, 0, np.inf)
        en = pen_wt * clipped * (ap > pen_wt)
        en /= (scad_param - 1) * pen_wt
        en += pen_wt * (ap <= pen_wt)
        en /= eps + ap
        hm.flat[::hm.shape[0] + 1] += self.num_group * en
        sn -= self.num_group * en * params
        try:
            update = np.linalg.solve(hm, sn)
        except np.linalg.LinAlgError:
            update = np.dot(np.linalg.pinv(hm), sn)
            msg = 'Encountered singularity in regularized GEE update'
            warnings.warn(msg)
        hm *= self.estimate_scale()
        return (update, hm)

    def _regularized_covmat(self, mean_params):
        self.update_cached_means(mean_params)
        ma = 0
        for i in range(self.num_group):
            expval, _ = self.cached_means[i]
            resid = self.endog_li[i] - expval
            sdev = np.sqrt(self.family.variance(expval))
            ex = self.exog_li[i] * sdev[:, None] ** 2
            rslt = self.cov_struct.covariance_matrix_solve(expval, i, sdev, (resid,))
            ma0 = np.dot(ex.T, rslt[0])
            ma += np.outer(ma0, ma0)
        return ma

    def fit_regularized(self, pen_wt, scad_param=3.7, maxiter=100, ddof_scale=None, update_assoc=5, ctol=1e-05, ztol=0.001, eps=1e-06, scale=None):
        """
        Regularized estimation for GEE.

        Parameters
        ----------
        pen_wt : float
            The penalty weight (a non-negative scalar).
        scad_param : float
            Non-negative scalar determining the shape of the Scad
            penalty.
        maxiter : int
            The maximum number of iterations.
        ddof_scale : int
            Value to subtract from `nobs` when calculating the
            denominator degrees of freedom for t-statistics, defaults
            to the number of columns in `exog`.
        update_assoc : int
            The dependence parameters are updated every `update_assoc`
            iterations of the mean structure parameter updates.
        ctol : float
            Convergence criterion, default is one order of magnitude
            smaller than proposed in section 3.1 of Wang et al.
        ztol : float
            Coefficients smaller than this value are treated as
            being zero, default is based on section 5 of Wang et al.
        eps : non-negative scalar
            Numerical constant, see section 3.2 of Wang et al.
        scale : float or string
            If a float, this value is used as the scale parameter.
            If "X2", the scale parameter is always estimated using
            Pearson's chi-square method (e.g. as in a quasi-Poisson
            analysis).  If None, the default approach for the family
            is used to estimate the scale parameter.

        Returns
        -------
        GEEResults instance.  Note that not all methods of the results
        class make sense when the model has been fit with regularization.

        Notes
        -----
        This implementation assumes that the link is canonical.

        References
        ----------
        Wang L, Zhou J, Qu A. (2012). Penalized generalized estimating
        equations for high-dimensional longitudinal data analysis.
        Biometrics. 2012 Jun;68(2):353-60.
        doi: 10.1111/j.1541-0420.2011.01678.x.
        https://www.ncbi.nlm.nih.gov/pubmed/21955051
        http://users.stat.umn.edu/~wangx346/research/GEE_selection.pdf
        """
        self.scaletype = scale
        mean_params = np.zeros(self.exog.shape[1])
        self.update_cached_means(mean_params)
        converged = False
        fit_history = defaultdict(list)
        if ddof_scale is None:
            self.ddof_scale = self.exog.shape[1]
        else:
            if not ddof_scale >= 0:
                raise ValueError('ddof_scale must be a non-negative number or None')
            self.ddof_scale = ddof_scale
        miniter = 20
        for itr in range(maxiter):
            update, hm = self._update_regularized(mean_params, pen_wt, scad_param, eps)
            if update is None:
                msg = 'Singular matrix encountered in regularized GEE update'
                warnings.warn(msg, ConvergenceWarning)
                break
            if itr > miniter and np.sqrt(np.sum(update ** 2)) < ctol:
                converged = True
                break
            mean_params += update
            fit_history['params'].append(mean_params.copy())
            self.update_cached_means(mean_params)
            if itr != 0 and itr % update_assoc == 0:
                self._update_assoc(mean_params)
        if not converged:
            msg = 'GEE.fit_regularized did not converge'
            warnings.warn(msg)
        mean_params[np.abs(mean_params) < ztol] = 0
        self._update_assoc(mean_params)
        ma = self._regularized_covmat(mean_params)
        cov = np.linalg.solve(hm, ma)
        cov = np.linalg.solve(hm, cov.T)
        res_kwds = dict(cov_type='robust', cov_robust=cov)
        scale = self.estimate_scale()
        rslt = GEEResults(self, mean_params, cov, scale, regularized=True, attr_kwds=res_kwds)
        rslt.fit_history = fit_history
        return GEEResultsWrapper(rslt)

    def _handle_constraint(self, mean_params, bcov):
        """
        Expand the parameter estimate `mean_params` and covariance matrix
        `bcov` to the coordinate system of the unconstrained model.

        Parameters
        ----------
        mean_params : array_like
            A parameter vector estimate for the reduced model.
        bcov : array_like
            The covariance matrix of mean_params.

        Returns
        -------
        mean_params : array_like
            The input parameter vector mean_params, expanded to the
            coordinate system of the full model
        bcov : array_like
            The input covariance matrix bcov, expanded to the
            coordinate system of the full model
        """
        red_p = len(mean_params)
        full_p = self.constraint.lhs.shape[1]
        mean_params0 = np.r_[mean_params, np.zeros(full_p - red_p)]
        save_exog_li = self.exog_li
        self.exog_li = self.constraint.exog_fulltrans_li
        import copy
        save_cached_means = copy.deepcopy(self.cached_means)
        self.update_cached_means(mean_params0)
        _, score = self._update_mean_params()
        if score is None:
            warnings.warn('Singular matrix encountered in GEE score test', ConvergenceWarning)
            return (None, None)
        _, ncov1, cmat = self._covmat()
        scale = self.estimate_scale()
        cmat = cmat / scale ** 2
        score2 = score[red_p:] / scale
        amat = np.linalg.inv(ncov1)
        bmat_11 = cmat[0:red_p, 0:red_p]
        bmat_22 = cmat[red_p:, red_p:]
        bmat_12 = cmat[0:red_p, red_p:]
        amat_11 = amat[0:red_p, 0:red_p]
        amat_12 = amat[0:red_p, red_p:]
        score_cov = bmat_22 - np.dot(amat_12.T, np.linalg.solve(amat_11, bmat_12))
        score_cov -= np.dot(bmat_12.T, np.linalg.solve(amat_11, amat_12))
        score_cov += np.dot(amat_12.T, np.dot(np.linalg.solve(amat_11, bmat_11), np.linalg.solve(amat_11, amat_12)))
        from scipy.stats.distributions import chi2
        score_statistic = np.dot(score2, np.linalg.solve(score_cov, score2))
        score_df = len(score2)
        score_pvalue = 1 - chi2.cdf(score_statistic, score_df)
        self.score_test_results = {'statistic': score_statistic, 'df': score_df, 'p-value': score_pvalue}
        mean_params = self.constraint.unpack_param(mean_params)
        bcov = self.constraint.unpack_cov(bcov)
        self.exog_li = save_exog_li
        self.cached_means = save_cached_means
        self.exog = self.constraint.restore_exog()
        return (mean_params, bcov)

    def _update_assoc(self, params):
        """
        Update the association parameters
        """
        self.cov_struct.update(params)

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None):
        """
        For computing marginal effects, returns dF(XB) / dX where F(.)
        is the fitted mean.

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        offset_exposure = None
        if exog is None:
            exog = self.exog
            offset_exposure = self._offset_exposure
        margeff = self.mean_deriv_exog(exog, params, offset_exposure)
        if 'ex' in transform:
            margeff *= exog
        if 'ey' in transform:
            margeff /= self.predict(params, exog)[:, None]
        if count_idx is not None:
            from statsmodels.discrete.discrete_margins import _get_count_effects
            margeff = _get_count_effects(margeff, exog, count_idx, transform, self, params)
        if dummy_idx is not None:
            from statsmodels.discrete.discrete_margins import _get_dummy_effects
            margeff = _get_dummy_effects(margeff, exog, dummy_idx, transform, self, params)
        return margeff

    def qic(self, params, scale, cov_params, n_step=1000):
        """
        Returns quasi-information criteria and quasi-likelihood values.

        Parameters
        ----------
        params : array_like
            The GEE estimates of the regression parameters.
        scale : scalar
            Estimated scale parameter
        cov_params : array_like
            An estimate of the covariance matrix for the
            model parameters.  Conventionally this is the robust
            covariance matrix.
        n_step : integer
            The number of points in the trapezoidal approximation
            to the quasi-likelihood function.

        Returns
        -------
        ql : scalar
            The quasi-likelihood value
        qic : scalar
            A QIC that can be used to compare the mean and covariance
            structures of the model.
        qicu : scalar
            A simplified QIC that can be used to compare mean structures
            but not covariance structures

        Notes
        -----
        The quasi-likelihood used here is obtained by numerically evaluating
        Wedderburn's integral representation of the quasi-likelihood function.
        This approach is valid for all families and  links.  Many other
        packages use analytical expressions for quasi-likelihoods that are
        valid in special cases where the link function is canonical.  These
        analytical expressions may omit additive constants that only depend
        on the data.  Therefore, the numerical values of our QL and QIC values
        will differ from the values reported by other packages.  However only
        the differences between two QIC values calculated for different models
        using the same data are meaningful.  Our QIC should produce the same
        QIC differences as other software.

        When using the QIC for models with unknown scale parameter, use a
        common estimate of the scale parameter for all models being compared.

        References
        ----------
        .. [*] W. Pan (2001).  Akaike's information criterion in generalized
               estimating equations.  Biometrics (57) 1.
        """
        varfunc = self.family.variance
        means = []
        omega = 0.0
        for i in range(self.num_group):
            expval, lpr = self.cached_means[i]
            means.append(expval)
            dmat = self.mean_deriv(self.exog_li[i], lpr)
            omega += np.dot(dmat.T, dmat) / scale
        means = np.concatenate(means)
        endog_li = np.concatenate(self.endog_li)
        du = means - endog_li
        qv = np.empty(n_step)
        xv = np.linspace(-0.99999, 1, n_step)
        for i, g in enumerate(xv):
            u = endog_li + (g + 1) * du / 2.0
            vu = varfunc(u)
            qv[i] = -np.sum(du ** 2 * (g + 1) / vu)
        qv /= 4 * scale
        try:
            from scipy.integrate import trapezoid
        except ImportError:
            from scipy.integrate import trapz as trapezoid
        ql = trapezoid(qv, dx=xv[1] - xv[0])
        qicu = -2 * ql + 2 * self.exog.shape[1]
        qic = -2 * ql + 2 * np.trace(np.dot(omega, cov_params))
        return (ql, qic, qicu)