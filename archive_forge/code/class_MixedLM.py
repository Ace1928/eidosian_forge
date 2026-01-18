import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
class MixedLM(base.LikelihoodModel):
    """
    Linear Mixed Effects Model

    Parameters
    ----------
    endog : 1d array_like
        The dependent variable
    exog : 2d array_like
        A matrix of covariates used to determine the
        mean structure (the "fixed effects" covariates).
    groups : 1d array_like
        A vector of labels determining the groups -- data from
        different groups are independent
    exog_re : 2d array_like
        A matrix of covariates used to determine the variance and
        covariance structure (the "random effects" covariates).  If
        None, defaults to a random intercept for each group.
    exog_vc : VCSpec instance or dict-like (deprecated)
        A VCSPec instance defines the structure of the variance
        components in the model.  Alternatively, see notes below
        for a dictionary-based format.  The dictionary format is
        deprecated and may be removed at some point in the future.
    use_sqrt : bool
        If True, optimization is carried out using the lower
        triangle of the square root of the random effects
        covariance matrix, otherwise it is carried out using the
        lower triangle of the random effects covariance matrix.
    missing : str
        The approach to missing data handling

    Notes
    -----
    If `exog_vc` is not a `VCSpec` instance, then it must be a
    dictionary of dictionaries.  Specifically, `exog_vc[a][g]` is a
    matrix whose columns are linearly combined using independent
    random coefficients.  This random term then contributes to the
    variance structure of the data for group `g`.  The random
    coefficients all have mean zero, and have the same variance.  The
    matrix must be `m x k`, where `m` is the number of observations in
    group `g`.  The number of columns may differ among the top-level
    groups.

    The covariates in `exog`, `exog_re` and `exog_vc` may (but need
    not) partially or wholly overlap.

    `use_sqrt` should almost always be set to True.  The main use case
    for use_sqrt=False is when complicated patterns of fixed values in
    the covariance structure are set (using the `free` argument to
    `fit`) that cannot be expressed in terms of the Cholesky factor L.

    Examples
    --------
    A basic mixed model with fixed effects for the columns of
    ``exog`` and a random intercept for each distinct value of
    ``group``:

    >>> model = sm.MixedLM(endog, exog, groups)
    >>> result = model.fit()

    A mixed model with fixed effects for the columns of ``exog`` and
    correlated random coefficients for the columns of ``exog_re``:

    >>> model = sm.MixedLM(endog, exog, groups, exog_re=exog_re)
    >>> result = model.fit()

    A mixed model with fixed effects for the columns of ``exog`` and
    independent random coefficients for the columns of ``exog_re``:

    >>> free = MixedLMParams.from_components(
                     fe_params=np.ones(exog.shape[1]),
                     cov_re=np.eye(exog_re.shape[1]))
    >>> model = sm.MixedLM(endog, exog, groups, exog_re=exog_re)
    >>> result = model.fit(free=free)

    A different way to specify independent random coefficients for the
    columns of ``exog_re``.  In this example ``groups`` must be a
    Pandas Series with compatible indexing with ``exog_re``, and
    ``exog_re`` has two columns.

    >>> g = pd.groupby(groups, by=groups).groups
    >>> vc = {}
    >>> vc['1'] = {k : exog_re.loc[g[k], 0] for k in g}
    >>> vc['2'] = {k : exog_re.loc[g[k], 1] for k in g}
    >>> model = sm.MixedLM(endog, exog, groups, vcomp=vc)
    >>> result = model.fit()
    """

    def __init__(self, endog, exog, groups, exog_re=None, exog_vc=None, use_sqrt=True, missing='none', **kwargs):
        _allowed_kwargs = ['missing_idx', 'design_info', 'formula']
        for x in kwargs.keys():
            if x not in _allowed_kwargs:
                raise ValueError('argument %s not permitted for MixedLM initialization' % x)
        self.use_sqrt = use_sqrt
        self.reml = True
        self.fe_pen = None
        self.re_pen = None
        if isinstance(exog_vc, dict):
            warnings.warn('Using deprecated variance components format')
            exog_vc = _convert_vc(exog_vc)
        if exog_vc is not None:
            self.k_vc = len(exog_vc.names)
            self.exog_vc = exog_vc
        else:
            self.k_vc = 0
            self.exog_vc = VCSpec([], [], [])
        if exog is not None and data_tools._is_using_ndarray_type(exog, None) and (exog.ndim == 1):
            exog = exog[:, None]
        if exog_re is not None and data_tools._is_using_ndarray_type(exog_re, None) and (exog_re.ndim == 1):
            exog_re = exog_re[:, None]
        super().__init__(endog, exog, groups=groups, exog_re=exog_re, missing=missing, **kwargs)
        self._init_keys.extend(['use_sqrt', 'exog_vc'])
        self.k_fe = exog.shape[1]
        if exog_re is None and len(self.exog_vc.names) == 0:
            self.k_re = 1
            self.k_re2 = 1
            self.exog_re = np.ones((len(endog), 1), dtype=np.float64)
            self.data.exog_re = self.exog_re
            names = ['Group Var']
            self.data.param_names = self.exog_names + names
            self.data.exog_re_names = names
            self.data.exog_re_names_full = names
        elif exog_re is not None:
            self.data.exog_re = exog_re
            self.exog_re = np.asarray(exog_re)
            if self.exog_re.ndim == 1:
                self.exog_re = self.exog_re[:, None]
            self.k_re = self.exog_re.shape[1]
            self.k_re2 = self.k_re * (self.k_re + 1) // 2
        else:
            self.k_re = 0
            self.k_re2 = 0
        if not self.data._param_names:
            param_names, exog_re_names, exog_re_names_full = self._make_param_names(exog_re)
            self.data.param_names = param_names
            self.data.exog_re_names = exog_re_names
            self.data.exog_re_names_full = exog_re_names_full
        self.k_params = self.k_fe + self.k_re2
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = {s: [] for s in group_labels}
        for i, g in enumerate(groups):
            row_indices[g].append(i)
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.n_groups = len(self.group_labels)
        self.endog_li = self.group_list(self.endog)
        self.exog_li = self.group_list(self.exog)
        self.exog_re_li = self.group_list(self.exog_re)
        if self.exog_re is None:
            self.exog_re2_li = None
        else:
            self.exog_re2_li = [np.dot(x.T, x) for x in self.exog_re_li]
        self.nobs = len(self.endog)
        self.n_totobs = self.nobs
        if self.exog_names is None:
            self.exog_names = ['FE%d' % (k + 1) for k in range(self.exog.shape[1])]
        self._aex_r = []
        self._aex_r2 = []
        for i in range(self.n_groups):
            a = self._augment_exog(i)
            self._aex_r.append(a)
            ma = _dot(a.T, a)
            self._aex_r2.append(ma)
        self._lin, self._quad = self._reparam()

    def _make_param_names(self, exog_re):
        """
        Returns the full parameter names list, just the exogenous random
        effects variables, and the exogenous random effects variables with
        the interaction terms.
        """
        exog_names = list(self.exog_names)
        exog_re_names = _get_exog_re_names(self, exog_re)
        param_names = []
        jj = self.k_fe
        for i in range(len(exog_re_names)):
            for j in range(i + 1):
                if i == j:
                    param_names.append(exog_re_names[i] + ' Var')
                else:
                    param_names.append(exog_re_names[j] + ' x ' + exog_re_names[i] + ' Cov')
                jj += 1
        vc_names = [x + ' Var' for x in self.exog_vc.names]
        return (exog_names + param_names + vc_names, exog_re_names, param_names)

    @classmethod
    def from_formula(cls, formula, data, re_formula=None, vc_formula=None, subset=None, use_sparse=False, missing='none', *args, **kwargs):
        """
        Create a Model from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        data : array_like
            The data for the model. See Notes.
        re_formula : str
            A one-sided formula defining the variance structure of the
            model.  The default gives a random intercept for each
            group.
        vc_formula : dict-like
            Formulas describing variance components.  `vc_formula[vc]` is
            the formula for the component with variance parameter named
            `vc`.  The formula is processed into a matrix, and the columns
            of this matrix are linearly combined with independent random
            coefficients having mean zero and a common variance.
        subset : array_like
            An array-like object of booleans, integers, or index
            values that indicate the subset of df to use in the
            model. Assumes df is a `pandas.DataFrame`
        missing : str
            Either 'none' or 'drop'
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model : Model instance

        Notes
        -----
        `data` must define __getitem__ with the keys in the formula
        terms args and kwargs are passed on to the model
        instantiation. E.g., a numpy structured or rec array, a
        dictionary, or a pandas DataFrame.

        If the variance component is intended to produce random
        intercepts for disjoint subsets of a group, specified by
        string labels or a categorical data value, always use '0 +' in
        the formula so that no overall intercept is included.

        If the variance components specify random slopes and you do
        not also want a random group-level intercept in the model,
        then use '0 +' in the formula to exclude the intercept.

        The variance components formulas are processed separately for
        each group.  If a variable is categorical the results will not
        be affected by whether the group labels are distinct or
        re-used over the top-level groups.

        Examples
        --------
        Suppose we have data from an educational study with students
        nested in classrooms nested in schools.  The students take a
        test, and we want to relate the test scores to the students'
        ages, while accounting for the effects of classrooms and
        schools.  The school will be the top-level group, and the
        classroom is a nested group that is specified as a variance
        component.  Note that the schools may have different number of
        classrooms, and the classroom labels may (but need not be)
        different across the schools.

        >>> vc = {'classroom': '0 + C(classroom)'}
        >>> MixedLM.from_formula('test_score ~ age', vc_formula=vc,                                   re_formula='1', groups='school', data=data)

        Now suppose we also have a previous test score called
        'pretest'.  If we want the relationship between pretest
        scores and the current test to vary by classroom, we can
        specify a random slope for the pretest score

        >>> vc = {'classroom': '0 + C(classroom)', 'pretest': '0 + pretest'}
        >>> MixedLM.from_formula('test_score ~ age + pretest', vc_formula=vc,                                   re_formula='1', groups='school', data=data)

        The following model is almost equivalent to the previous one,
        but here the classroom random intercept and pretest slope may
        be correlated.

        >>> vc = {'classroom': '0 + C(classroom)'}
        >>> MixedLM.from_formula('test_score ~ age + pretest', vc_formula=vc,                                   re_formula='1 + pretest', groups='school',                                   data=data)
        """
        if 'groups' not in kwargs.keys():
            raise AttributeError("'groups' is a required keyword argument " + 'in MixedLM.from_formula')
        groups = kwargs['groups']
        group_name = 'Group'
        if isinstance(groups, str):
            group_name = groups
            groups = np.asarray(data[groups])
        else:
            groups = np.asarray(groups)
        del kwargs['groups']
        if missing == 'drop':
            data, groups = _handle_missing(data, groups, formula, re_formula, vc_formula)
            missing = 'none'
        if re_formula is not None:
            if re_formula.strip() == '1':
                exog_re = np.ones((data.shape[0], 1))
                exog_re_names = [group_name]
            else:
                eval_env = kwargs.get('eval_env', None)
                if eval_env is None:
                    eval_env = 1
                elif eval_env == -1:
                    from patsy import EvalEnvironment
                    eval_env = EvalEnvironment({})
                exog_re = patsy.dmatrix(re_formula, data, eval_env=eval_env)
                exog_re_names = exog_re.design_info.column_names
                exog_re_names = [x.replace('Intercept', group_name) for x in exog_re_names]
                exog_re = np.asarray(exog_re)
            if exog_re.ndim == 1:
                exog_re = exog_re[:, None]
        else:
            exog_re = None
            if vc_formula is None:
                exog_re_names = [group_name]
            else:
                exog_re_names = []
        if vc_formula is not None:
            eval_env = kwargs.get('eval_env', None)
            if eval_env is None:
                eval_env = 1
            elif eval_env == -1:
                from patsy import EvalEnvironment
                eval_env = EvalEnvironment({})
            vc_mats = []
            vc_colnames = []
            vc_names = []
            gb = data.groupby(groups)
            kylist = sorted(gb.groups.keys())
            vcf = sorted(vc_formula.keys())
            for vc_name in vcf:
                md = patsy.ModelDesc.from_formula(vc_formula[vc_name])
                vc_names.append(vc_name)
                evc_mats, evc_colnames = ([], [])
                for group_ix, group in enumerate(kylist):
                    ii = gb.groups[group]
                    mat = patsy.dmatrix(md, data.loc[ii, :], eval_env=eval_env, return_type='dataframe')
                    evc_colnames.append(mat.columns.tolist())
                    if use_sparse:
                        evc_mats.append(sparse.csr_matrix(mat))
                    else:
                        evc_mats.append(np.asarray(mat))
                vc_mats.append(evc_mats)
                vc_colnames.append(evc_colnames)
            exog_vc = VCSpec(vc_names, vc_colnames, vc_mats)
        else:
            exog_vc = VCSpec([], [], [])
        kwargs['subset'] = None
        kwargs['exog_re'] = exog_re
        kwargs['exog_vc'] = exog_vc
        kwargs['groups'] = groups
        mod = super().from_formula(formula, data, *args, **kwargs)
        param_names, exog_re_names, exog_re_names_full = mod._make_param_names(exog_re_names)
        mod.data.param_names = param_names
        mod.data.exog_re_names = exog_re_names
        mod.data.exog_re_names_full = exog_re_names_full
        if vc_formula is not None:
            mod.data.vcomp_names = mod.exog_vc.names
        return mod

    def predict(self, params, exog=None):
        """
        Return predicted values from a design matrix.

        Parameters
        ----------
        params : array_like
            Parameters of a mixed linear model.  Can be either a
            MixedLMParams instance, or a vector containing the packed
            model parameters in which the fixed effects parameters are
            at the beginning of the vector, or a vector containing
            only the fixed effects parameters.
        exog : array_like, optional
            Design / exogenous data for the fixed effects. Model exog
            is used if None.

        Returns
        -------
        An array of fitted values.  Note that these predicted values
        only reflect the fixed effects mean structure of the model.
        """
        if exog is None:
            exog = self.exog
        if isinstance(params, MixedLMParams):
            params = params.fe_params
        else:
            params = params[0:self.k_fe]
        return np.dot(exog, params)

    def group_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        grouping structure.
        """
        if array is None:
            return None
        if array.ndim == 1:
            return [np.array(array[self.row_indices[k]]) for k in self.group_labels]
        else:
            return [np.array(array[self.row_indices[k], :]) for k in self.group_labels]

    def fit_regularized(self, start_params=None, method='l1', alpha=0, ceps=0.0001, ptol=1e-06, maxit=200, **fit_kwargs):
        """
        Fit a model in which the fixed effects parameters are
        penalized.  The dependence parameters are held fixed at their
        estimated values in the unpenalized model.

        Parameters
        ----------
        method : str of Penalty object
            Method for regularization.  If a string, must be 'l1'.
        alpha : array_like
            Scalar or vector of penalty weights.  If a scalar, the
            same weight is applied to all coefficients; if a vector,
            it contains a weight for each coefficient.  If method is a
            Penalty object, the weights are scaled by alpha.  For L1
            regularization, the weights are used directly.
        ceps : positive real scalar
            Fixed effects parameters smaller than this value
            in magnitude are treated as being zero.
        ptol : positive real scalar
            Convergence occurs when the sup norm difference
            between successive values of `fe_params` is less than
            `ptol`.
        maxit : int
            The maximum number of iterations.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        A MixedLMResults instance containing the results.

        Notes
        -----
        The covariance structure is not updated as the fixed effects
        parameters are varied.

        The algorithm used here for L1 regularization is a"shooting"
        or cyclic coordinate descent algorithm.

        If method is 'l1', then `fe_pen` and `cov_pen` are used to
        obtain the covariance structure, but are ignored during the
        L1-penalized fitting.

        References
        ----------
        Friedman, J. H., Hastie, T. and Tibshirani, R. Regularized
        Paths for Generalized Linear Models via Coordinate
        Descent. Journal of Statistical Software, 33(1) (2008)
        http://www.jstatsoft.org/v33/i01/paper

        http://statweb.stanford.edu/~tibs/stat315a/Supplements/fuse.pdf
        """
        if isinstance(method, str) and method.lower() != 'l1':
            raise ValueError('Invalid regularization method')
        if isinstance(method, Penalty):
            method.alpha = alpha
            fit_kwargs.update({'fe_pen': method})
            return self.fit(**fit_kwargs)
        if np.isscalar(alpha):
            alpha = alpha * np.ones(self.k_fe, dtype=np.float64)
        mdf = self.fit(**fit_kwargs)
        fe_params = mdf.fe_params
        cov_re = mdf.cov_re
        vcomp = mdf.vcomp
        scale = mdf.scale
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None
        for itr in range(maxit):
            fe_params_s = fe_params.copy()
            for j in range(self.k_fe):
                if abs(fe_params[j]) < ceps:
                    continue
                fe_params[j] = 0.0
                expval = np.dot(self.exog, fe_params)
                resid_all = self.endog - expval
                a, b = (0.0, 0.0)
                for group_ix, group in enumerate(self.group_labels):
                    vc_var = self._expand_vcomp(vcomp, group_ix)
                    exog = self.exog_li[group_ix]
                    ex_r, ex2_r = (self._aex_r[group_ix], self._aex_r2[group_ix])
                    resid = resid_all[self.row_indices[group]]
                    solver = _smw_solver(scale, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
                    x = exog[:, j]
                    u = solver(x)
                    a += np.dot(u, x)
                    b -= 2 * np.dot(u, resid)
                pwt1 = alpha[j]
                if b > pwt1:
                    fe_params[j] = -(b - pwt1) / (2 * a)
                elif b < -pwt1:
                    fe_params[j] = -(b + pwt1) / (2 * a)
            if np.abs(fe_params_s - fe_params).max() < ptol:
                break
        params_prof = mdf.params.copy()
        params_prof[0:self.k_fe] = fe_params
        scale = self.get_scale(fe_params, mdf.cov_re_unscaled, mdf.vcomp)
        hess, sing = self.hessian(params_prof)
        if sing:
            warnings.warn(_warn_cov_sing)
        pcov = np.nan * np.ones_like(hess)
        ii = np.abs(params_prof) > ceps
        ii[self.k_fe:] = True
        ii = np.flatnonzero(ii)
        hess1 = hess[ii, :][:, ii]
        pcov[np.ix_(ii, ii)] = np.linalg.inv(-hess1)
        params_object = MixedLMParams.from_components(fe_params, cov_re=cov_re)
        results = MixedLMResults(self, params_prof, pcov / scale)
        results.params_object = params_object
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.vcomp = vcomp
        results.scale = scale
        results.cov_re_unscaled = mdf.cov_re_unscaled
        results.method = mdf.method
        results.converged = True
        results.cov_pen = self.cov_pen
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2
        results.k_vc = self.k_vc
        return MixedLMResultsWrapper(results)

    def get_fe_params(self, cov_re, vcomp, tol=1e-10):
        """
        Use GLS to update the fixed effects parameter estimates.

        Parameters
        ----------
        cov_re : array_like (2d)
            The covariance matrix of the random effects.
        vcomp : array_like (1d)
            The variance components.
        tol : float
            A tolerance parameter to determine when covariances
            are singular.

        Returns
        -------
        params : ndarray
            The GLS estimates of the fixed effects parameters.
        singular : bool
            True if the covariance is singular
        """
        if self.k_fe == 0:
            return (np.array([]), False)
        sing = False
        if self.k_re == 0:
            cov_re_inv = np.empty((0, 0))
        else:
            w, v = np.linalg.eigh(cov_re)
            if w.min() < tol:
                sing = True
                ii = np.flatnonzero(w >= tol)
                if len(ii) == 0:
                    cov_re_inv = np.zeros_like(cov_re)
                else:
                    vi = v[:, ii]
                    wi = w[ii]
                    cov_re_inv = np.dot(vi / wi, vi.T)
            else:
                cov_re_inv = np.linalg.inv(cov_re)
        if not hasattr(self, '_endex_li'):
            self._endex_li = []
            for group_ix, _ in enumerate(self.group_labels):
                mat = np.concatenate((self.exog_li[group_ix], self.endog_li[group_ix][:, None]), axis=1)
                self._endex_li.append(mat)
        xtxy = 0.0
        for group_ix, group in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            if vc_var.size > 0:
                if vc_var.min() < tol:
                    sing = True
                    ii = np.flatnonzero(vc_var >= tol)
                    vc_vari = np.zeros_like(vc_var)
                    vc_vari[ii] = 1 / vc_var[ii]
                else:
                    vc_vari = 1 / vc_var
            else:
                vc_vari = np.empty(0)
            exog = self.exog_li[group_ix]
            ex_r, ex2_r = (self._aex_r[group_ix], self._aex_r2[group_ix])
            solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, vc_vari)
            u = solver(self._endex_li[group_ix])
            xtxy += np.dot(exog.T, u)
        if sing:
            fe_params = np.dot(np.linalg.pinv(xtxy[:, 0:-1]), xtxy[:, -1])
        else:
            fe_params = np.linalg.solve(xtxy[:, 0:-1], xtxy[:, -1])
        return (fe_params, sing)

    def _reparam(self):
        """
        Returns parameters of the map converting parameters from the
        form used in optimization to the form returned to the user.

        Returns
        -------
        lin : list-like
            Linear terms of the map
        quad : list-like
            Quadratic terms of the map

        Notes
        -----
        If P are the standard form parameters and R are the
        transformed parameters (i.e. with the Cholesky square root
        covariance and square root transformed variance components),
        then P[i] = lin[i] * R + R' * quad[i] * R
        """
        k_fe, k_re, k_re2, k_vc = (self.k_fe, self.k_re, self.k_re2, self.k_vc)
        k_tot = k_fe + k_re2 + k_vc
        ix = np.tril_indices(self.k_re)
        lin = []
        for k in range(k_fe):
            e = np.zeros(k_tot)
            e[k] = 1
            lin.append(e)
        for k in range(k_re2):
            lin.append(np.zeros(k_tot))
        for k in range(k_vc):
            lin.append(np.zeros(k_tot))
        quad = []
        for k in range(k_tot):
            quad.append(np.zeros((k_tot, k_tot)))
        ii = np.tril_indices(k_re)
        ix = [(a, b) for a, b in zip(ii[0], ii[1])]
        for i1 in range(k_re2):
            for i2 in range(k_re2):
                ix1 = ix[i1]
                ix2 = ix[i2]
                if ix1[1] == ix2[1] and ix1[0] <= ix2[0]:
                    ii = (ix2[0], ix1[0])
                    k = ix.index(ii)
                    quad[k_fe + k][k_fe + i2, k_fe + i1] += 1
        for k in range(k_tot):
            quad[k] = 0.5 * (quad[k] + quad[k].T)
        km = k_fe + k_re2
        for k in range(km, km + k_vc):
            quad[k][k, k] = 1
        return (lin, quad)

    def _expand_vcomp(self, vcomp, group_ix):
        """
        Replicate variance parameters to match a group's design.

        Parameters
        ----------
        vcomp : array_like
            The variance parameters for the variance components.
        group_ix : int
            The group index

        Returns an expanded version of vcomp, in which each variance
        parameter is copied as many times as there are independent
        realizations of the variance component in the given group.
        """
        if len(vcomp) == 0:
            return np.empty(0)
        vc_var = []
        for j in range(len(self.exog_vc.names)):
            d = self.exog_vc.mats[j][group_ix].shape[1]
            vc_var.append(vcomp[j] * np.ones(d))
        if len(vc_var) > 0:
            return np.concatenate(vc_var)
        else:
            return np.empty(0)

    def _augment_exog(self, group_ix):
        """
        Concatenate the columns for variance components to the columns
        for other random effects to obtain a single random effects
        exog matrix for a given group.
        """
        ex_r = self.exog_re_li[group_ix] if self.k_re > 0 else None
        if self.k_vc == 0:
            return ex_r
        ex = [ex_r] if self.k_re > 0 else []
        any_sparse = False
        for j, _ in enumerate(self.exog_vc.names):
            ex.append(self.exog_vc.mats[j][group_ix])
            any_sparse |= sparse.issparse(ex[-1])
        if any_sparse:
            for j, x in enumerate(ex):
                if not sparse.issparse(x):
                    ex[j] = sparse.csr_matrix(x)
            ex = sparse.hstack(ex)
            ex = sparse.csr_matrix(ex)
        else:
            ex = np.concatenate(ex, axis=1)
        return ex

    def loglike(self, params, profile_fe=True):
        """
        Evaluate the (profile) log-likelihood of the linear mixed
        effects model.

        Parameters
        ----------
        params : MixedLMParams, or array_like.
            The parameter value.  If array-like, must be a packed
            parameter vector containing only the covariance
            parameters.
        profile_fe : bool
            If True, replace the provided value of `fe_params` with
            the GLS estimates.

        Returns
        -------
        The log-likelihood value at `params`.

        Notes
        -----
        The scale parameter `scale` is always profiled out of the
        log-likelihood.  In addition, if `profile_fe` is true the
        fixed effects parameters are also profiled out.
        """
        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe, self.k_re, self.use_sqrt, has_fe=False)
        cov_re = params.cov_re
        vcomp = params.vcomp
        if profile_fe:
            fe_params, sing = self.get_fe_params(cov_re, vcomp)
            if sing:
                self._cov_sing += 1
        else:
            fe_params = params.fe_params
        if self.k_re > 0:
            try:
                cov_re_inv = np.linalg.inv(cov_re)
            except np.linalg.LinAlgError:
                cov_re_inv = np.linalg.pinv(cov_re)
                self._cov_sing += 1
            _, cov_re_logdet = np.linalg.slogdet(cov_re)
        else:
            cov_re_inv = np.zeros((0, 0))
            cov_re_logdet = 0
        expval = np.dot(self.exog, fe_params)
        resid_all = self.endog - expval
        likeval = 0.0
        if self.cov_pen is not None and self.k_re > 0:
            likeval -= self.cov_pen.func(cov_re, cov_re_inv)
        if self.fe_pen is not None:
            likeval -= self.fe_pen.func(fe_params)
        xvx, qf = (0.0, 0.0)
        for group_ix, group in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            cov_aug_logdet = cov_re_logdet + np.sum(np.log(vc_var))
            exog = self.exog_li[group_ix]
            ex_r, ex2_r = (self._aex_r[group_ix], self._aex_r2[group_ix])
            solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
            resid = resid_all[self.row_indices[group]]
            ld = _smw_logdet(1.0, ex_r, ex2_r, cov_re_inv, 1 / vc_var, cov_aug_logdet)
            likeval -= ld / 2.0
            u = solver(resid)
            qf += np.dot(resid, u)
            if self.reml:
                mat = solver(exog)
                xvx += np.dot(exog.T, mat)
        if self.reml:
            likeval -= (self.n_totobs - self.k_fe) * np.log(qf) / 2.0
            _, ld = np.linalg.slogdet(xvx)
            likeval -= ld / 2.0
            likeval -= (self.n_totobs - self.k_fe) * np.log(2 * np.pi) / 2.0
            likeval += (self.n_totobs - self.k_fe) * np.log(self.n_totobs - self.k_fe) / 2.0
            likeval -= (self.n_totobs - self.k_fe) / 2.0
        else:
            likeval -= self.n_totobs * np.log(qf) / 2.0
            likeval -= self.n_totobs * np.log(2 * np.pi) / 2.0
            likeval += self.n_totobs * np.log(self.n_totobs) / 2.0
            likeval -= self.n_totobs / 2.0
        return likeval

    def _gen_dV_dPar(self, ex_r, solver, group_ix, max_ix=None):
        """
        A generator that yields the element-wise derivative of the
        marginal covariance matrix with respect to the random effects
        variance and covariance parameters.

        ex_r : array_like
            The random effects design matrix
        solver : function
            A function that given x returns V^{-1}x, where V
            is the group's marginal covariance matrix.
        group_ix : int
            The group index
        max_ix : {int, None}
            If not None, the generator ends when this index
            is reached.
        """
        axr = solver(ex_r)
        jj = 0
        for j1 in range(self.k_re):
            for j2 in range(j1 + 1):
                if max_ix is not None and jj > max_ix:
                    return
                mat_l, mat_r = (ex_r[:, j1:j1 + 1], ex_r[:, j2:j2 + 1])
                vsl, vsr = (axr[:, j1:j1 + 1], axr[:, j2:j2 + 1])
                yield (jj, mat_l, mat_r, vsl, vsr, j1 == j2)
                jj += 1
        for j, _ in enumerate(self.exog_vc.names):
            if max_ix is not None and jj > max_ix:
                return
            mat = self.exog_vc.mats[j][group_ix]
            axmat = solver(mat)
            yield (jj, mat, mat, axmat, axmat, True)
            jj += 1

    def score(self, params, profile_fe=True):
        """
        Returns the score vector of the profile log-likelihood.

        Notes
        -----
        The score vector that is returned is computed with respect to
        the parameterization defined by this model instance's
        `use_sqrt` attribute.
        """
        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe, self.k_re, self.use_sqrt, has_fe=False)
        if profile_fe:
            params.fe_params, sing = self.get_fe_params(params.cov_re, params.vcomp)
            if sing:
                msg = 'Random effects covariance is singular'
                warnings.warn(msg)
        if self.use_sqrt:
            score_fe, score_re, score_vc = self.score_sqrt(params, calc_fe=not profile_fe)
        else:
            score_fe, score_re, score_vc = self.score_full(params, calc_fe=not profile_fe)
        if self._freepat is not None:
            score_fe *= self._freepat.fe_params
            score_re *= self._freepat.cov_re[self._freepat._ix]
            score_vc *= self._freepat.vcomp
        if profile_fe:
            return np.concatenate((score_re, score_vc))
        else:
            return np.concatenate((score_fe, score_re, score_vc))

    def score_full(self, params, calc_fe):
        """
        Returns the score with respect to untransformed parameters.

        Calculates the score vector for the profiled log-likelihood of
        the mixed effects model with respect to the parameterization
        in which the random effects covariance matrix is represented
        in its full form (not using the Cholesky factor).

        Parameters
        ----------
        params : MixedLMParams or array_like
            The parameter at which the score function is evaluated.
            If array-like, must contain the packed random effects
            parameters (cov_re and vcomp) without fe_params.
        calc_fe : bool
            If True, calculate the score vector for the fixed effects
            parameters.  If False, this vector is not calculated, and
            a vector of zeros is returned in its place.

        Returns
        -------
        score_fe : array_like
            The score vector with respect to the fixed effects
            parameters.
        score_re : array_like
            The score vector with respect to the random effects
            parameters (excluding variance components parameters).
        score_vc : array_like
            The score vector with respect to variance components
            parameters.

        Notes
        -----
        `score_re` is taken with respect to the parameterization in
        which `cov_re` is represented through its lower triangle
        (without taking the Cholesky square root).
        """
        fe_params = params.fe_params
        cov_re = params.cov_re
        vcomp = params.vcomp
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = np.linalg.pinv(cov_re)
            self._cov_sing += 1
        score_fe = np.zeros(self.k_fe)
        score_re = np.zeros(self.k_re2)
        score_vc = np.zeros(self.k_vc)
        if self.cov_pen is not None:
            score_re -= self.cov_pen.deriv(cov_re, cov_re_inv)
        if calc_fe and self.fe_pen is not None:
            score_fe -= self.fe_pen.deriv(fe_params)
        rvir = 0.0
        xtvir = 0.0
        xtvix = 0.0
        xtax = [0.0] * (self.k_re2 + self.k_vc)
        dlv = np.zeros(self.k_re2 + self.k_vc)
        rvavr = np.zeros(self.k_re2 + self.k_vc)
        for group_ix, group in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            exog = self.exog_li[group_ix]
            ex_r, ex2_r = (self._aex_r[group_ix], self._aex_r2[group_ix])
            solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
            resid = self.endog_li[group_ix]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval
            if self.reml:
                viexog = solver(exog)
                xtvix += np.dot(exog.T, viexog)
            vir = solver(resid)
            for jj, matl, matr, vsl, vsr, sym in self._gen_dV_dPar(ex_r, solver, group_ix):
                dlv[jj] = _dotsum(matr, vsl)
                if not sym:
                    dlv[jj] += _dotsum(matl, vsr)
                ul = _dot(vir, matl)
                ur = ul.T if sym else _dot(matr.T, vir)
                ulr = np.dot(ul, ur)
                rvavr[jj] += ulr
                if not sym:
                    rvavr[jj] += ulr.T
                if self.reml:
                    ul = _dot(viexog.T, matl)
                    ur = ul.T if sym else _dot(matr.T, viexog)
                    ulr = np.dot(ul, ur)
                    xtax[jj] += ulr
                    if not sym:
                        xtax[jj] += ulr.T
            if self.k_re > 0:
                score_re -= 0.5 * dlv[0:self.k_re2]
            if self.k_vc > 0:
                score_vc -= 0.5 * dlv[self.k_re2:]
            rvir += np.dot(resid, vir)
            if calc_fe:
                xtvir += np.dot(exog.T, vir)
        fac = self.n_totobs
        if self.reml:
            fac -= self.k_fe
        if calc_fe and self.k_fe > 0:
            score_fe += fac * xtvir / rvir
        if self.k_re > 0:
            score_re += 0.5 * fac * rvavr[0:self.k_re2] / rvir
        if self.k_vc > 0:
            score_vc += 0.5 * fac * rvavr[self.k_re2:] / rvir
        if self.reml:
            xtvixi = np.linalg.inv(xtvix)
            for j in range(self.k_re2):
                score_re[j] += 0.5 * _dotsum(xtvixi.T, xtax[j])
            for j in range(self.k_vc):
                score_vc[j] += 0.5 * _dotsum(xtvixi.T, xtax[self.k_re2 + j])
        return (score_fe, score_re, score_vc)

    def score_sqrt(self, params, calc_fe=True):
        """
        Returns the score with respect to transformed parameters.

        Calculates the score vector with respect to the
        parameterization in which the random effects covariance matrix
        is represented through its Cholesky square root.

        Parameters
        ----------
        params : MixedLMParams or array_like
            The model parameters.  If array-like must contain packed
            parameters that are compatible with this model instance.
        calc_fe : bool
            If True, calculate the score vector for the fixed effects
            parameters.  If False, this vector is not calculated, and
            a vector of zeros is returned in its place.

        Returns
        -------
        score_fe : array_like
            The score vector with respect to the fixed effects
            parameters.
        score_re : array_like
            The score vector with respect to the random effects
            parameters (excluding variance components parameters).
        score_vc : array_like
            The score vector with respect to variance components
            parameters.
        """
        score_fe, score_re, score_vc = self.score_full(params, calc_fe=calc_fe)
        params_vec = params.get_packed(use_sqrt=True, has_fe=True)
        score_full = np.concatenate((score_fe, score_re, score_vc))
        scr = 0.0
        for i in range(len(params_vec)):
            v = self._lin[i] + 2 * np.dot(self._quad[i], params_vec)
            scr += score_full[i] * v
        score_fe = scr[0:self.k_fe]
        score_re = scr[self.k_fe:self.k_fe + self.k_re2]
        score_vc = scr[self.k_fe + self.k_re2:]
        return (score_fe, score_re, score_vc)

    def hessian(self, params):
        """
        Returns the model's Hessian matrix.

        Calculates the Hessian matrix for the linear mixed effects
        model with respect to the parameterization in which the
        covariance matrix is represented directly (without square-root
        transformation).

        Parameters
        ----------
        params : MixedLMParams or array_like
            The model parameters at which the Hessian is calculated.
            If array-like, must contain the packed parameters in a
            form that is compatible with this model instance.

        Returns
        -------
        hess : 2d ndarray
            The Hessian matrix, evaluated at `params`.
        sing : boolean
            If True, the covariance matrix is singular and a
            pseudo-inverse is returned.
        """
        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe, self.k_re, use_sqrt=self.use_sqrt, has_fe=True)
        fe_params = params.fe_params
        vcomp = params.vcomp
        cov_re = params.cov_re
        sing = False
        if self.k_re > 0:
            try:
                cov_re_inv = np.linalg.inv(cov_re)
            except np.linalg.LinAlgError:
                cov_re_inv = np.linalg.pinv(cov_re)
                sing = True
        else:
            cov_re_inv = np.empty((0, 0))
        hess_fe = 0.0
        hess_re = np.zeros((self.k_re2 + self.k_vc, self.k_re2 + self.k_vc))
        hess_fere = np.zeros((self.k_re2 + self.k_vc, self.k_fe))
        fac = self.n_totobs
        if self.reml:
            fac -= self.exog.shape[1]
        rvir = 0.0
        xtvix = 0.0
        xtax = [0.0] * (self.k_re2 + self.k_vc)
        m = self.k_re2 + self.k_vc
        B = np.zeros(m)
        D = np.zeros((m, m))
        F = [[0.0] * m for k in range(m)]
        for group_ix, group in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            vc_vari = np.zeros_like(vc_var)
            ii = np.flatnonzero(vc_var >= 1e-10)
            if len(ii) > 0:
                vc_vari[ii] = 1 / vc_var[ii]
            if len(ii) < len(vc_var):
                sing = True
            exog = self.exog_li[group_ix]
            ex_r, ex2_r = (self._aex_r[group_ix], self._aex_r2[group_ix])
            solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, vc_vari)
            resid = self.endog_li[group_ix]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval
            viexog = solver(exog)
            xtvix += np.dot(exog.T, viexog)
            vir = solver(resid)
            rvir += np.dot(resid, vir)
            for jj1, matl1, matr1, vsl1, vsr1, sym1 in self._gen_dV_dPar(ex_r, solver, group_ix):
                ul = _dot(viexog.T, matl1)
                ur = _dot(matr1.T, vir)
                hess_fere[jj1, :] += np.dot(ul, ur)
                if not sym1:
                    ul = _dot(viexog.T, matr1)
                    ur = _dot(matl1.T, vir)
                    hess_fere[jj1, :] += np.dot(ul, ur)
                if self.reml:
                    ul = _dot(viexog.T, matl1)
                    ur = ul if sym1 else np.dot(viexog.T, matr1)
                    ulr = _dot(ul, ur.T)
                    xtax[jj1] += ulr
                    if not sym1:
                        xtax[jj1] += ulr.T
                ul = _dot(vir, matl1)
                ur = ul if sym1 else _dot(vir, matr1)
                B[jj1] += np.dot(ul, ur) * (1 if sym1 else 2)
                E = [(vsl1, matr1)]
                if not sym1:
                    E.append((vsr1, matl1))
                for jj2, matl2, matr2, vsl2, vsr2, sym2 in self._gen_dV_dPar(ex_r, solver, group_ix, jj1):
                    re = sum([_multi_dot_three(matr2.T, x[0], x[1].T) for x in E])
                    vt = 2 * _dot(_multi_dot_three(vir[None, :], matl2, re), vir[:, None])
                    if not sym2:
                        le = sum([_multi_dot_three(matl2.T, x[0], x[1].T) for x in E])
                        vt += 2 * _dot(_multi_dot_three(vir[None, :], matr2, le), vir[:, None])
                    D[jj1, jj2] += np.squeeze(vt)
                    if jj1 != jj2:
                        D[jj2, jj1] += np.squeeze(vt)
                    rt = _dotsum(vsl2, re.T) / 2
                    if not sym2:
                        rt += _dotsum(vsr2, le.T) / 2
                    hess_re[jj1, jj2] += rt
                    if jj1 != jj2:
                        hess_re[jj2, jj1] += rt
                    if self.reml:
                        ev = sum([_dot(x[0], _dot(x[1].T, viexog)) for x in E])
                        u1 = _dot(viexog.T, matl2)
                        u2 = _dot(matr2.T, ev)
                        um = np.dot(u1, u2)
                        F[jj1][jj2] += um + um.T
                        if not sym2:
                            u1 = np.dot(viexog.T, matr2)
                            u2 = np.dot(matl2.T, ev)
                            um = np.dot(u1, u2)
                            F[jj1][jj2] += um + um.T
        hess_fe -= fac * xtvix / rvir
        hess_re = hess_re - 0.5 * fac * (D / rvir - np.outer(B, B) / rvir ** 2)
        hess_fere = -fac * hess_fere / rvir
        if self.reml:
            QL = [np.linalg.solve(xtvix, x) for x in xtax]
            for j1 in range(self.k_re2 + self.k_vc):
                for j2 in range(j1 + 1):
                    a = _dotsum(QL[j1].T, QL[j2])
                    a -= np.trace(np.linalg.solve(xtvix, F[j1][j2]))
                    a *= 0.5
                    hess_re[j1, j2] += a
                    if j1 > j2:
                        hess_re[j2, j1] += a
        m = self.k_fe + self.k_re2 + self.k_vc
        hess = np.zeros((m, m))
        hess[0:self.k_fe, 0:self.k_fe] = hess_fe
        hess[0:self.k_fe, self.k_fe:] = hess_fere.T
        hess[self.k_fe:, 0:self.k_fe] = hess_fere
        hess[self.k_fe:, self.k_fe:] = hess_re
        return (hess, sing)

    def get_scale(self, fe_params, cov_re, vcomp):
        """
        Returns the estimated error variance based on given estimates
        of the slopes and random effects covariance matrix.

        Parameters
        ----------
        fe_params : array_like
            The regression slope estimates
        cov_re : 2d array_like
            Estimate of the random effects covariance matrix
        vcomp : array_like
            Estimate of the variance components

        Returns
        -------
        scale : float
            The estimated error variance.
        """
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = np.linalg.pinv(cov_re)
            warnings.warn(_warn_cov_sing)
        qf = 0.0
        for group_ix, group in enumerate(self.group_labels):
            vc_var = self._expand_vcomp(vcomp, group_ix)
            exog = self.exog_li[group_ix]
            ex_r, ex2_r = (self._aex_r[group_ix], self._aex_r2[group_ix])
            solver = _smw_solver(1.0, ex_r, ex2_r, cov_re_inv, 1 / vc_var)
            resid = self.endog_li[group_ix]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval
            mat = solver(resid)
            qf += np.dot(resid, mat)
        if self.reml:
            qf /= self.n_totobs - self.k_fe
        else:
            qf /= self.n_totobs
        return qf

    def fit(self, start_params=None, reml=True, niter_sa=0, do_cg=True, fe_pen=None, cov_pen=None, free=None, full_output=False, method=None, **fit_kwargs):
        """
        Fit a linear mixed model to the data.

        Parameters
        ----------
        start_params : array_like or MixedLMParams
            Starting values for the profile log-likelihood.  If not a
            `MixedLMParams` instance, this should be an array
            containing the packed parameters for the profile
            log-likelihood, including the fixed effects
            parameters.
        reml : bool
            If true, fit according to the REML likelihood, else
            fit the standard likelihood using ML.
        niter_sa : int
            Currently this argument is ignored and has no effect
            on the results.
        cov_pen : CovariancePenalty object
            A penalty for the random effects covariance matrix
        do_cg : bool, defaults to True
            If False, the optimization is skipped and a results
            object at the given (or default) starting values is
            returned.
        fe_pen : Penalty object
            A penalty on the fixed effects
        free : MixedLMParams object
            If not `None`, this is a mask that allows parameters to be
            held fixed at specified values.  A 1 indicates that the
            corresponding parameter is estimated, a 0 indicates that
            it is fixed at its starting value.  Setting the `cov_re`
            component to the identity matrix fits a model with
            independent random effects.  Note that some optimization
            methods do not respect this constraint (bfgs and lbfgs both
            work).
        full_output : bool
            If true, attach iteration history to results
        method : str
            Optimization method.  Can be a scipy.optimize method name,
            or a list of such names to be tried in sequence.
        **fit_kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        A MixedLMResults instance.
        """
        _allowed_kwargs = ['gtol', 'maxiter', 'eps', 'maxcor', 'ftol', 'tol', 'disp', 'maxls']
        for x in fit_kwargs.keys():
            if x not in _allowed_kwargs:
                warnings.warn('Argument %s not used by MixedLM.fit' % x)
        if method is None:
            method = ['bfgs', 'lbfgs', 'cg']
        elif isinstance(method, str):
            method = [method]
        for meth in method:
            if meth.lower() in ['newton', 'ncg']:
                raise ValueError('method %s not available for MixedLM' % meth)
        self.reml = reml
        self.cov_pen = cov_pen
        self.fe_pen = fe_pen
        self._cov_sing = 0
        self._freepat = free
        if full_output:
            hist = []
        else:
            hist = None
        if start_params is None:
            params = MixedLMParams(self.k_fe, self.k_re, self.k_vc)
            params.fe_params = np.zeros(self.k_fe)
            params.cov_re = np.eye(self.k_re)
            params.vcomp = np.ones(self.k_vc)
        elif isinstance(start_params, MixedLMParams):
            params = start_params
        elif len(start_params) == self.k_fe + self.k_re2 + self.k_vc:
            params = MixedLMParams.from_packed(start_params, self.k_fe, self.k_re, self.use_sqrt, has_fe=True)
        elif len(start_params) == self.k_re2 + self.k_vc:
            params = MixedLMParams.from_packed(start_params, self.k_fe, self.k_re, self.use_sqrt, has_fe=False)
        else:
            raise ValueError('invalid start_params')
        if do_cg:
            fit_kwargs['retall'] = hist is not None
            if 'disp' not in fit_kwargs:
                fit_kwargs['disp'] = False
            packed = params.get_packed(use_sqrt=self.use_sqrt, has_fe=False)
            if niter_sa > 0:
                warnings.warn('niter_sa is currently ignored')
            for j in range(len(method)):
                rslt = super().fit(start_params=packed, skip_hessian=True, method=method[j], **fit_kwargs)
                if rslt.mle_retvals['converged']:
                    break
                packed = rslt.params
                if j + 1 < len(method):
                    next_method = method[j + 1]
                    warnings.warn('Retrying MixedLM optimization with %s' % next_method, ConvergenceWarning)
                else:
                    msg = 'MixedLM optimization failed, ' + 'trying a different optimizer may help.'
                    warnings.warn(msg, ConvergenceWarning)
            params = np.atleast_1d(rslt.params)
            if hist is not None:
                hist.append(rslt.mle_retvals)
        converged = rslt.mle_retvals['converged']
        if not converged:
            gn = self.score(rslt.params)
            gn = np.sqrt(np.sum(gn ** 2))
            msg = 'Gradient optimization failed, |grad| = %f' % gn
            warnings.warn(msg, ConvergenceWarning)
        params = MixedLMParams.from_packed(params, self.k_fe, self.k_re, use_sqrt=self.use_sqrt, has_fe=False)
        cov_re_unscaled = params.cov_re
        vcomp_unscaled = params.vcomp
        fe_params, sing = self.get_fe_params(cov_re_unscaled, vcomp_unscaled)
        params.fe_params = fe_params
        scale = self.get_scale(fe_params, cov_re_unscaled, vcomp_unscaled)
        cov_re = scale * cov_re_unscaled
        vcomp = scale * vcomp_unscaled
        f1 = self.k_re > 0 and np.min(np.abs(np.diag(cov_re))) < 0.01
        f2 = self.k_vc > 0 and np.min(np.abs(vcomp)) < 0.01
        if f1 or f2:
            msg = 'The MLE may be on the boundary of the parameter space.'
            warnings.warn(msg, ConvergenceWarning)
        hess, sing = self.hessian(params)
        if sing:
            warnings.warn(_warn_cov_sing)
        hess_diag = np.diag(hess)
        if free is not None:
            pcov = np.zeros_like(hess)
            pat = self._freepat.get_packed(use_sqrt=False, has_fe=True)
            ii = np.flatnonzero(pat)
            hess_diag = hess_diag[ii]
            if len(ii) > 0:
                hess1 = hess[np.ix_(ii, ii)]
                pcov[np.ix_(ii, ii)] = np.linalg.inv(-hess1)
        else:
            pcov = np.linalg.inv(-hess)
        if np.any(hess_diag >= 0):
            msg = 'The Hessian matrix at the estimated parameter values ' + 'is not positive definite.'
            warnings.warn(msg, ConvergenceWarning)
        params_packed = params.get_packed(use_sqrt=False, has_fe=True)
        results = MixedLMResults(self, params_packed, pcov / scale)
        results.params_object = params
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.vcomp = vcomp
        results.scale = scale
        results.cov_re_unscaled = cov_re_unscaled
        results.method = 'REML' if self.reml else 'ML'
        results.converged = converged
        results.hist = hist
        results.reml = self.reml
        results.cov_pen = self.cov_pen
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2
        results.k_vc = self.k_vc
        results.use_sqrt = self.use_sqrt
        results.freepat = self._freepat
        return MixedLMResultsWrapper(results)

    def get_distribution(self, params, scale, exog):
        return _mixedlm_distribution(self, params, scale, exog)