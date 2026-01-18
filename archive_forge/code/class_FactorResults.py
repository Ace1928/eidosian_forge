import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
class FactorResults:
    """
    Factor results class

    For result summary, scree/loading plots and factor rotations

    Parameters
    ----------
    factor : Factor
        Fitted Factor class

    Attributes
    ----------
    uniqueness : ndarray
        The uniqueness (variance of uncorrelated errors unique to
        each variable)
    communality : ndarray
        1 - uniqueness
    loadings : ndarray
        Each column is the loading vector for one factor
    loadings_no_rot : ndarray
        Unrotated loadings, not available under maximum likelihood
        analysis.
    eigenvals : ndarray
        The eigenvalues for a factor analysis obtained using
        principal components; not available under ML estimation.
    n_comp : int
        Number of components (factors)
    nbs : int
        Number of observations
    fa_method : str
        The method used to obtain the decomposition, either 'pa' for
        'principal axes' or 'ml' for maximum likelihood.
    df : int
        Degrees of freedom of the factor model.

    Notes
    -----
    Under ML estimation, the default rotation (used for `loadings`) is
    condition IC3 of Bai and Li (2012).  Under this rotation, the
    factor scores are iid and standardized.  If `G` is the canonical
    loadings and `U` is the vector of uniquenesses, then the
    covariance matrix implied by the factor analysis is `GG' +
    diag(U)`.

    Status: experimental, Some refactoring will be necessary when new
        features are added.
    """

    def __init__(self, factor):
        self.model = factor
        self.endog_names = factor.endog_names
        self.loadings_no_rot = factor.loadings
        if hasattr(factor, 'eigenvals'):
            self.eigenvals = factor.eigenvals
        self.communality = factor.communality
        self.uniqueness = factor.uniqueness
        self.rotation_method = None
        self.fa_method = factor.method
        self.n_comp = factor.loadings.shape[1]
        self.nobs = factor.nobs
        self._factor = factor
        if hasattr(factor, 'mle_retvals'):
            self.mle_retvals = factor.mle_retvals
        p, k = self.loadings_no_rot.shape
        self.df = ((p - k) ** 2 - (p + k)) // 2
        self.loadings = factor.loadings
        self.rotation_matrix = np.eye(self.n_comp)

    def __str__(self):
        return self.summary().__str__()

    def rotate(self, method):
        """
        Apply rotation, inplace modification of this Results instance

        Parameters
        ----------
        method : str
            Rotation to be applied.  Allowed methods are varimax,
            quartimax, biquartimax, equamax, oblimin, parsimax,
            parsimony, biquartimin, promax.

        Returns
        -------
        None : nothing returned, modifications are inplace


        Notes
        -----
        Warning: 'varimax', 'quartimax' and 'oblimin' are verified against R or
        Stata. Some rotation methods such as promax do not produce the same
        results as the R or Stata default functions.

        See Also
        --------
        factor_rotation : subpackage that implements rotation methods
        """
        self.rotation_method = method
        if method not in ['varimax', 'quartimax', 'biquartimax', 'equamax', 'oblimin', 'parsimax', 'parsimony', 'biquartimin', 'promax']:
            raise ValueError('Unknown rotation method %s' % method)
        if method in ['varimax', 'quartimax', 'biquartimax', 'equamax', 'parsimax', 'parsimony', 'biquartimin']:
            self.loadings, T = rotate_factors(self.loadings_no_rot, method)
        elif method == 'oblimin':
            self.loadings, T = rotate_factors(self.loadings_no_rot, 'quartimin')
        elif method == 'promax':
            self.loadings, T = promax(self.loadings_no_rot)
        else:
            raise ValueError('rotation method not recognized')
        self.rotation_matrix = T

    def _corr_factors(self):
        """correlation of factors implied by rotation

        If the rotation is oblique, then the factors are correlated.

        currently not cached

        Returns
        -------
        corr_f : ndarray
            correlation matrix of rotated factors, assuming initial factors are
            orthogonal
        """
        T = self.rotation_matrix
        corr_f = T.T.dot(T)
        return corr_f

    def factor_score_params(self, method='bartlett'):
        """
        Compute factor scoring coefficient matrix

        The coefficient matrix is not cached.

        Parameters
        ----------
        method : 'bartlett' or 'regression'
            Method to use for factor scoring.
            'regression' can be abbreviated to `reg`

        Returns
        -------
        coeff_matrix : ndarray
            matrix s to compute factors f from a standardized endog ys.
            ``f = ys dot s``

        Notes
        -----
        The `regression` method follows the Stata definition.
        Method bartlett and regression are verified against Stats.
        Two unofficial methods, 'ols' and 'gls', produce similar factor scores
        but are not verified.

        See Also
        --------
        statsmodels.multivariate.factor.FactorResults.factor_scoring
        """
        L = self.loadings
        T = self.rotation_matrix.T
        uni = 1 - self.communality
        if method == 'bartlett':
            s_mat = np.linalg.inv(L.T.dot(L / uni[:, None])).dot(L.T / uni).T
        elif method.startswith('reg'):
            corr = self.model.corr
            corr_f = self._corr_factors()
            s_mat = corr_f.dot(L.T.dot(np.linalg.inv(corr))).T
        elif method == 'ols':
            corr = self.model.corr
            corr_f = self._corr_factors()
            s_mat = corr_f.dot(np.linalg.pinv(L)).T
        elif method == 'gls':
            corr = self.model.corr
            corr_f = self._corr_factors()
            s_mat = np.linalg.inv(np.linalg.inv(corr_f) + L.T.dot(L / uni[:, None]))
            s_mat = s_mat.dot(L.T / uni).T
        else:
            raise ValueError('method not available, use "bartlett ' + 'or "regression"')
        return s_mat

    def factor_scoring(self, endog=None, method='bartlett', transform=True):
        """
        factor scoring: compute factors for endog

        If endog was not provided when creating the factor class, then
        a standarized endog needs to be provided here.

        Parameters
        ----------
        method : 'bartlett' or 'regression'
            Method to use for factor scoring.
            'regression' can be abbreviated to `reg`
        transform : bool
            If transform is true and endog is provided, then it will be
            standardized using mean and scale of original data, which has to
            be available in this case.
            If transform is False, then a provided endog will be used unchanged.
            The original endog in the Factor class will
            always be standardized if endog is None, independently of `transform`.

        Returns
        -------
        factor_score : ndarray
            estimated factors using scoring matrix s and standarized endog ys
            ``f = ys dot s``

        Notes
        -----
        Status: transform option is experimental and might change.

        See Also
        --------
        statsmodels.multivariate.factor.FactorResults.factor_score_params
        """
        if transform is False and endog is not None:
            endog = np.asarray(endog)
        else:
            if self.model.endog is not None:
                m = self.model.endog.mean(0)
                s = self.model.endog.std(ddof=1, axis=0)
                if endog is None:
                    endog = self.model.endog
                else:
                    endog = np.asarray(endog)
            else:
                raise ValueError('If transform is True, then `endog` needs ' + 'to be available in the Factor instance.')
            endog = (endog - m) / s
        s_mat = self.factor_score_params(method=method)
        factors = endog.dot(s_mat)
        return factors

    def summary(self):
        """Summary"""
        summ = summary2.Summary()
        summ.add_title('Factor analysis results')
        loadings_no_rot = pd.DataFrame(self.loadings_no_rot, columns=['factor %d' % i for i in range(self.loadings_no_rot.shape[1])], index=self.endog_names)
        if hasattr(self, 'eigenvals'):
            eigenvals = pd.DataFrame([self.eigenvals], columns=self.endog_names, index=[''])
            summ.add_dict({'': 'Eigenvalues'})
            summ.add_df(eigenvals)
        communality = pd.DataFrame([self.communality], columns=self.endog_names, index=[''])
        summ.add_dict({'': ''})
        summ.add_dict({'': 'Communality'})
        summ.add_df(communality)
        summ.add_dict({'': ''})
        summ.add_dict({'': 'Pre-rotated loadings'})
        summ.add_df(loadings_no_rot)
        summ.add_dict({'': ''})
        if self.rotation_method is not None:
            loadings = pd.DataFrame(self.loadings, columns=['factor %d' % i for i in range(self.loadings.shape[1])], index=self.endog_names)
            summ.add_dict({'': '%s rotated loadings' % self.rotation_method})
            summ.add_df(loadings)
        return summ

    def get_loadings_frame(self, style='display', sort_=True, threshold=0.3, highlight_max=True, color_max='yellow', decimals=None):
        """get loadings matrix as DataFrame or pandas Styler

        Parameters
        ----------
        style : 'display' (default), 'raw' or 'strings'
            Style to use for display

            * 'raw' returns just a DataFrame of the loadings matrix, no options are
               applied
            * 'display' add sorting and styling as defined by other keywords
            * 'strings' returns a DataFrame with string elements with optional sorting
               and suppressing small loading coefficients.

        sort_ : bool
            If True, then the rows of the DataFrame is sorted by contribution of each
            factor. applies if style is either 'display' or 'strings'
        threshold : float
            If the threshold is larger than zero, then loading coefficients are
            either colored white (if style is 'display') or replace by empty
            string (if style is 'strings').
        highlight_max : bool
            This add a background color to the largest coefficient in each row.
        color_max : html color
            default is 'yellow'. color for background of row maximum
        decimals : None or int
            If None, then pandas default precision applies. Otherwise values are
            rounded to the specified decimals. If style is 'display', then the
            underlying dataframe is not changed. If style is 'strings', then
            values are rounded before conversion to strings.

        Returns
        -------
        loadings : DataFrame or pandas Styler instance
            The return is a pandas Styler instance, if style is 'display' and
            at least one of highlight_max, threshold or decimals is applied.
            Otherwise, the returned loadings is a DataFrame.

        Examples
        --------
        >>> mod = Factor(df, 3, smc=True)
        >>> res = mod.fit()
        >>> res.get_loadings_frame(style='display', decimals=3, threshold=0.2)

        To get a sorted DataFrame, all styling options need to be turned off:

        >>> df_sorted = res.get_loadings_frame(style='display',
        ...             highlight_max=False, decimals=None, threshold=0)

        Options except for highlighting are available for plain test or Latex
        usage:

        >>> lds = res_u.get_loadings_frame(style='strings', decimals=3,
        ...                                threshold=0.3)
        >>> print(lds.to_latex())
        """
        loadings_df = pd.DataFrame(self.loadings, columns=['factor %d' % i for i in range(self.loadings.shape[1])], index=self.endog_names)
        if style not in ['raw', 'display', 'strings']:
            msg = "style has to be one of 'raw', 'display', 'strings'"
            raise ValueError(msg)
        if style == 'raw':
            return loadings_df
        if sort_ is True:
            loadings_df2 = loadings_df.copy()
            n_f = len(loadings_df2)
            high = np.abs(loadings_df2.values).argmax(1)
            loadings_df2['high'] = high
            loadings_df2['largest'] = np.abs(loadings_df.values[np.arange(n_f), high])
            loadings_df2.sort_values(by=['high', 'largest'], ascending=[True, False], inplace=True)
            loadings_df = loadings_df2.drop(['high', 'largest'], axis=1)
        if style == 'display':
            sty = None
            if threshold > 0:

                def color_white_small(val):
                    """
                    Takes a scalar and returns a string with
                    the css property `'color: white'` for small values, black otherwise.

                    takes threshold from outer scope
                    """
                    color = 'white' if np.abs(val) < threshold else 'black'
                    return 'color: %s' % color
                try:
                    sty = loadings_df.style.map(color_white_small)
                except AttributeError:
                    sty = loadings_df.style.applymap(color_white_small)
            if highlight_max is True:

                def highlight_max(s):
                    """
                    highlight the maximum in a Series yellow.
                    """
                    s = np.abs(s)
                    is_max = s == s.max()
                    return ['background-color: ' + color_max if v else '' for v in is_max]
                if sty is None:
                    sty = loadings_df.style
                sty = sty.apply(highlight_max, axis=1)
            if decimals is not None:
                if sty is None:
                    sty = loadings_df.style
                sty.format('{:.%sf}' % decimals)
            if sty is None:
                return loadings_df
            else:
                return sty
        if style == 'strings':
            ld = loadings_df
            if decimals is not None:
                ld = ld.round(decimals)
            ld = ld.astype(str)
            if threshold > 0:
                ld[loadings_df.abs() < threshold] = ''
            return ld

    def plot_scree(self, ncomp=None):
        """
        Plot of the ordered eigenvalues and variance explained for the loadings

        Parameters
        ----------
        ncomp : int, optional
            Number of loadings to include in the plot.  If None, will
            included the same as the number of maximum possible loadings

        Returns
        -------
        Figure
            Handle to the figure.
        """
        _import_mpl()
        from .plots import plot_scree
        return plot_scree(self.eigenvals, self.n_comp, ncomp)

    def plot_loadings(self, loading_pairs=None, plot_prerotated=False):
        """
        Plot factor loadings in 2-d plots

        Parameters
        ----------
        loading_pairs : None or a list of tuples
            Specify plots. Each tuple (i, j) represent one figure, i and j is
            the loading number for x-axis and y-axis, respectively. If `None`,
            all combinations of the loadings will be plotted.
        plot_prerotated : True or False
            If True, the loadings before rotation applied will be plotted. If
            False, rotated loadings will be plotted.

        Returns
        -------
        figs : a list of figure handles
        """
        _import_mpl()
        from .plots import plot_loadings
        if self.rotation_method is None:
            plot_prerotated = True
        loadings = self.loadings_no_rot if plot_prerotated else self.loadings
        if plot_prerotated:
            title = 'Prerotated Factor Pattern'
        else:
            title = '%s Rotated Factor Pattern' % self.rotation_method
        var_explained = self.eigenvals / self.n_comp * 100
        return plot_loadings(loadings, loading_pairs=loading_pairs, title=title, row_names=self.endog_names, percent_variance=var_explained)

    @cache_readonly
    def fitted_cov(self):
        """
        Returns the fitted covariance matrix.
        """
        c = np.dot(self.loadings, self.loadings.T)
        c.flat[::c.shape[0] + 1] += self.uniqueness
        return c

    @cache_readonly
    def uniq_stderr(self, kurt=0):
        """
        The standard errors of the uniquenesses.

        Parameters
        ----------
        kurt : float
            Excess kurtosis

        Notes
        -----
        If excess kurtosis is known, provide as `kurt`.  Standard
        errors are only available if the model was fit using maximum
        likelihood.  If `endog` is not provided, `nobs` must be
        provided to obtain standard errors.

        These are asymptotic standard errors.  See Bai and Li (2012)
        for conditions under which the standard errors are valid.

        The standard errors are only applicable to the original,
        unrotated maximum likelihood solution.
        """
        if self.fa_method.lower() != 'ml':
            msg = 'Standard errors only available under ML estimation'
            raise ValueError(msg)
        if self.nobs is None:
            msg = 'nobs is required to obtain standard errors.'
            raise ValueError(msg)
        v = self.uniqueness ** 2 * (2 + kurt)
        return np.sqrt(v / self.nobs)

    @cache_readonly
    def load_stderr(self):
        """
        The standard errors of the loadings.

        Standard errors are only available if the model was fit using
        maximum likelihood.  If `endog` is not provided, `nobs` must be
        provided to obtain standard errors.

        These are asymptotic standard errors.  See Bai and Li (2012)
        for conditions under which the standard errors are valid.

        The standard errors are only applicable to the original,
        unrotated maximum likelihood solution.
        """
        if self.fa_method.lower() != 'ml':
            msg = 'Standard errors only available under ML estimation'
            raise ValueError(msg)
        if self.nobs is None:
            msg = 'nobs is required to obtain standard errors.'
            raise ValueError(msg)
        v = np.outer(self.uniqueness, np.ones(self.loadings.shape[1]))
        return np.sqrt(v / self.nobs)