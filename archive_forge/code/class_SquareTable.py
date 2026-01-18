import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly
class SquareTable(Table):
    """
    Methods for analyzing a square contingency table.

    Parameters
    ----------
    table : array_like
        A square contingency table, or DataFrame that is converted
        to a square form.
    shift_zeros : bool
        If True and any cell count is zero, add 0.5 to all values
        in the table.

    Notes
    -----
    These methods should only be used when the rows and columns of the
    table have the same categories.  If `table` is provided as a
    Pandas DataFrame, the row and column indices will be extended to
    create a square table, inserting zeros where a row or column is
    missing.  Otherwise the table should be provided in a square form,
    with the (implicit) row and column categories appearing in the
    same order.
    """

    def __init__(self, table, shift_zeros=True):
        table = _make_df_square(table)
        k1, k2 = table.shape
        if k1 != k2:
            raise ValueError('table must be square')
        super().__init__(table, shift_zeros)

    def symmetry(self, method='bowker'):
        """
        Test for symmetry of a joint distribution.

        This procedure tests the null hypothesis that the joint
        distribution is symmetric around the main diagonal, that is

        .. math::

        p_{i, j} = p_{j, i}  for all i, j

        Returns
        -------
        Bunch
            A bunch with attributes

            * statistic : float
                chisquare test statistic
            * p-value : float
                p-value of the test statistic based on chisquare distribution
            * df : int
                degrees of freedom of the chisquare distribution

        Notes
        -----
        The implementation is based on the SAS documentation. R includes
        it in `mcnemar.test` if the table is not 2 by 2.  However a more
        direct generalization of the McNemar test to larger tables is
        provided by the homogeneity test (TableSymmetry.homogeneity).

        The p-value is based on the chi-square distribution which requires
        that the sample size is not very small to be a good approximation
        of the true distribution. For 2x2 contingency tables the exact
        distribution can be obtained with `mcnemar`

        See Also
        --------
        mcnemar
        homogeneity
        """
        if method.lower() != 'bowker':
            raise ValueError("method for symmetry testing must be 'bowker'")
        k = self.table.shape[0]
        upp_idx = np.triu_indices(k, 1)
        tril = self.table.T[upp_idx]
        triu = self.table[upp_idx]
        statistic = ((tril - triu) ** 2 / (tril + triu + 1e-20)).sum()
        df = k * (k - 1) / 2.0
        pvalue = stats.chi2.sf(statistic, df)
        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue
        b.df = df
        return b

    def homogeneity(self, method='stuart_maxwell'):
        """
        Compare row and column marginal distributions.

        Parameters
        ----------
        method : str
            Either 'stuart_maxwell' or 'bhapkar', leading to two different
            estimates of the covariance matrix for the estimated
            difference between the row margins and the column margins.

        Returns
        -------
        Bunch
            A bunch with attributes:

            * statistic : float
                The chi^2 test statistic
            * pvalue : float
                The p-value of the test statistic
            * df : int
                The degrees of freedom of the reference distribution

        Notes
        -----
        For a 2x2 table this is equivalent to McNemar's test.  More
        generally the procedure tests the null hypothesis that the
        marginal distribution of the row factor is equal to the
        marginal distribution of the column factor.  For this to be
        meaningful, the two factors must have the same sample space
        (i.e. the same categories).
        """
        if self.table.shape[0] < 1:
            raise ValueError('table is empty')
        elif self.table.shape[0] == 1:
            b = _Bunch()
            b.statistic = 0
            b.pvalue = 1
            b.df = 0
            return b
        method = method.lower()
        if method not in ['bhapkar', 'stuart_maxwell']:
            raise ValueError("method '%s' for homogeneity not known" % method)
        n_obs = self.table.sum()
        pr = self.table.astype(np.float64) / n_obs
        row = pr.sum(1)[0:-1]
        col = pr.sum(0)[0:-1]
        pr = pr[0:-1, 0:-1]
        d = col - row
        df = pr.shape[0]
        if method == 'bhapkar':
            vmat = -(pr + pr.T) - np.outer(d, d)
            dv = col + row - 2 * np.diag(pr) - d ** 2
            np.fill_diagonal(vmat, dv)
        elif method == 'stuart_maxwell':
            vmat = -(pr + pr.T)
            dv = row + col - 2 * np.diag(pr)
            np.fill_diagonal(vmat, dv)
        try:
            statistic = n_obs * np.dot(d, np.linalg.solve(vmat, d))
        except np.linalg.LinAlgError:
            warnings.warn('Unable to invert covariance matrix', sm_exceptions.SingularMatrixWarning)
            b = _Bunch()
            b.statistic = np.nan
            b.pvalue = np.nan
            b.df = df
            return b
        pvalue = 1 - stats.chi2.cdf(statistic, df)
        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue
        b.df = df
        return b

    def summary(self, alpha=0.05, float_format='%.3f'):
        """
        Produce a summary of the analysis.

        Parameters
        ----------
        alpha : float
            `1 - alpha` is the nominal coverage probability of the interval.
        float_format : str
            Used to format numeric values in the table.
        method : str
            The method for producing the confidence interval.  Currently
            must be 'normal' which uses the normal approximation.
        """
        fmt = float_format
        headers = ['Statistic', 'P-value', 'DF']
        stubs = ['Symmetry', 'Homogeneity']
        sy = self.symmetry()
        hm = self.homogeneity()
        data = [[fmt % sy.statistic, fmt % sy.pvalue, '%d' % sy.df], [fmt % hm.statistic, fmt % hm.pvalue, '%d' % hm.df]]
        tab = iolib.SimpleTable(data, headers, stubs, data_aligns='r', table_dec_above='')
        return tab