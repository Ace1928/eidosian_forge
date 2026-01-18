import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
class random_table_gen(multi_rv_generic):
    """Contingency tables from independent samples with fixed marginal sums.

    This is the distribution of random tables with given row and column vector
    sums. This distribution represents the set of random tables under the null
    hypothesis that rows and columns are independent. It is used in hypothesis
    tests of independence.

    Because of assumed independence, the expected frequency of each table
    element can be computed from the row and column sums, so that the
    distribution is completely determined by these two vectors.

    Methods
    -------
    logpmf(x)
        Log-probability of table `x` to occur in the distribution.
    pmf(x)
        Probability of table `x` to occur in the distribution.
    mean(row, col)
        Mean table.
    rvs(row, col, size=None, method=None, random_state=None)
        Draw random tables with given row and column vector sums.

    Parameters
    ----------
    %(_doc_row_col)s
    %(_doc_random_state)s

    Notes
    -----
    %(_doc_row_col_note)s

    Random elements from the distribution are generated either with Boyett's
    [1]_ or Patefield's algorithm [2]_. Boyett's algorithm has
    O(N) time and space complexity, where N is the total sum of entries in the
    table. Patefield's algorithm has O(K x log(N)) time complexity, where K is
    the number of cells in the table and requires only a small constant work
    space. By default, the `rvs` method selects the fastest algorithm based on
    the input, but you can specify the algorithm with the keyword `method`.
    Allowed values are "boyett" and "patefield".

    .. versionadded:: 1.10.0

    Examples
    --------
    >>> from scipy.stats import random_table

    >>> row = [1, 5]
    >>> col = [2, 3, 1]
    >>> random_table.mean(row, col)
    array([[0.33333333, 0.5       , 0.16666667],
           [1.66666667, 2.5       , 0.83333333]])

    Alternatively, the object may be called (as a function) to fix the row
    and column vector sums, returning a "frozen" distribution.

    >>> dist = random_table(row, col)
    >>> dist.rvs(random_state=123)
    array([[1., 0., 0.],
           [1., 3., 1.]])

    References
    ----------
    .. [1] J. Boyett, AS 144 Appl. Statist. 28 (1979) 329-332
    .. [2] W.M. Patefield, AS 159 Appl. Statist. 30 (1981) 91-97
    """

    def __init__(self, seed=None):
        super().__init__(seed)

    def __call__(self, row, col, *, seed=None):
        """Create a frozen distribution of tables with given marginals.

        See `random_table_frozen` for more information.
        """
        return random_table_frozen(row, col, seed=seed)

    def logpmf(self, x, row, col):
        """Log-probability of table to occur in the distribution.

        Parameters
        ----------
        %(_doc_x)s
        %(_doc_row_col)s

        Returns
        -------
        logpmf : ndarray or scalar
            Log of the probability mass function evaluated at `x`.

        Notes
        -----
        %(_doc_row_col_note)s

        If row and column marginals of `x` do not match `row` and `col`,
        negative infinity is returned.

        Examples
        --------
        >>> from scipy.stats import random_table
        >>> import numpy as np

        >>> x = [[1, 5, 1], [2, 3, 1]]
        >>> row = np.sum(x, axis=1)
        >>> col = np.sum(x, axis=0)
        >>> random_table.logpmf(x, row, col)
        -1.6306401200847027

        Alternatively, the object may be called (as a function) to fix the row
        and column vector sums, returning a "frozen" distribution.

        >>> d = random_table(row, col)
        >>> d.logpmf(x)
        -1.6306401200847027
        """
        r, c, n = self._process_parameters(row, col)
        x = np.asarray(x)
        if x.ndim < 2:
            raise ValueError('`x` must be at least two-dimensional')
        dtype_is_int = np.issubdtype(x.dtype, np.integer)
        with np.errstate(invalid='ignore'):
            if not dtype_is_int and (not np.all(x.astype(int) == x)):
                raise ValueError('`x` must contain only integral values')
        if np.any(x < 0):
            raise ValueError('`x` must contain only non-negative values')
        r2 = np.sum(x, axis=-1)
        c2 = np.sum(x, axis=-2)
        if r2.shape[-1] != len(r):
            raise ValueError('shape of `x` must agree with `row`')
        if c2.shape[-1] != len(c):
            raise ValueError('shape of `x` must agree with `col`')
        res = np.empty(x.shape[:-2])
        mask = np.all(r2 == r, axis=-1) & np.all(c2 == c, axis=-1)

        def lnfac(x):
            return gammaln(x + 1)
        res[mask] = np.sum(lnfac(r), axis=-1) + np.sum(lnfac(c), axis=-1) - lnfac(n) - np.sum(lnfac(x[mask]), axis=(-1, -2))
        res[~mask] = -np.inf
        return res[()]

    def pmf(self, x, row, col):
        """Probability of table to occur in the distribution.

        Parameters
        ----------
        %(_doc_x)s
        %(_doc_row_col)s

        Returns
        -------
        pmf : ndarray or scalar
            Probability mass function evaluated at `x`.

        Notes
        -----
        %(_doc_row_col_note)s

        If row and column marginals of `x` do not match `row` and `col`,
        zero is returned.

        Examples
        --------
        >>> from scipy.stats import random_table
        >>> import numpy as np

        >>> x = [[1, 5, 1], [2, 3, 1]]
        >>> row = np.sum(x, axis=1)
        >>> col = np.sum(x, axis=0)
        >>> random_table.pmf(x, row, col)
        0.19580419580419592

        Alternatively, the object may be called (as a function) to fix the row
        and column vector sums, returning a "frozen" distribution.

        >>> d = random_table(row, col)
        >>> d.pmf(x)
        0.19580419580419592
        """
        return np.exp(self.logpmf(x, row, col))

    def mean(self, row, col):
        """Mean of distribution of conditional tables.
        %(_doc_mean_params)s

        Returns
        -------
        mean: ndarray
            Mean of the distribution.

        Notes
        -----
        %(_doc_row_col_note)s

        Examples
        --------
        >>> from scipy.stats import random_table

        >>> row = [1, 5]
        >>> col = [2, 3, 1]
        >>> random_table.mean(row, col)
        array([[0.33333333, 0.5       , 0.16666667],
               [1.66666667, 2.5       , 0.83333333]])

        Alternatively, the object may be called (as a function) to fix the row
        and column vector sums, returning a "frozen" distribution.

        >>> d = random_table(row, col)
        >>> d.mean()
        array([[0.33333333, 0.5       , 0.16666667],
               [1.66666667, 2.5       , 0.83333333]])
        """
        r, c, n = self._process_parameters(row, col)
        return np.outer(r, c) / n

    def rvs(self, row, col, *, size=None, method=None, random_state=None):
        """Draw random tables with fixed column and row marginals.

        Parameters
        ----------
        %(_doc_row_col)s
        size : integer, optional
            Number of samples to draw (default 1).
        method : str, optional
            Which method to use, "boyett" or "patefield". If None (default),
            selects the fastest method for this input.
        %(_doc_random_state)s

        Returns
        -------
        rvs : ndarray
            Random 2D tables of shape (`size`, `len(row)`, `len(col)`).

        Notes
        -----
        %(_doc_row_col_note)s

        Examples
        --------
        >>> from scipy.stats import random_table

        >>> row = [1, 5]
        >>> col = [2, 3, 1]
        >>> random_table.rvs(row, col, random_state=123)
        array([[1., 0., 0.],
               [1., 3., 1.]])

        Alternatively, the object may be called (as a function) to fix the row
        and column vector sums, returning a "frozen" distribution.

        >>> d = random_table(row, col)
        >>> d.rvs(random_state=123)
        array([[1., 0., 0.],
               [1., 3., 1.]])
        """
        r, c, n = self._process_parameters(row, col)
        size, shape = self._process_size_shape(size, r, c)
        random_state = self._get_random_state(random_state)
        meth = self._process_rvs_method(method, r, c, n)
        return meth(r, c, n, size, random_state).reshape(shape)

    @staticmethod
    def _process_parameters(row, col):
        """
        Check that row and column vectors are one-dimensional, that they do
        not contain negative or non-integer entries, and that the sums over
        both vectors are equal.
        """
        r = np.array(row, dtype=np.int64, copy=True)
        c = np.array(col, dtype=np.int64, copy=True)
        if np.ndim(r) != 1:
            raise ValueError('`row` must be one-dimensional')
        if np.ndim(c) != 1:
            raise ValueError('`col` must be one-dimensional')
        if np.any(r < 0):
            raise ValueError('each element of `row` must be non-negative')
        if np.any(c < 0):
            raise ValueError('each element of `col` must be non-negative')
        n = np.sum(r)
        if n != np.sum(c):
            raise ValueError('sums over `row` and `col` must be equal')
        if not np.all(r == np.asarray(row)):
            raise ValueError('each element of `row` must be an integer')
        if not np.all(c == np.asarray(col)):
            raise ValueError('each element of `col` must be an integer')
        return (r, c, n)

    @staticmethod
    def _process_size_shape(size, r, c):
        """
        Compute the number of samples to be drawn and the shape of the output
        """
        shape = (len(r), len(c))
        if size is None:
            return (1, shape)
        size = np.atleast_1d(size)
        if not np.issubdtype(size.dtype, np.integer) or np.any(size < 0):
            raise ValueError('`size` must be a non-negative integer or `None`')
        return (np.prod(size), tuple(size) + shape)

    @classmethod
    def _process_rvs_method(cls, method, r, c, n):
        known_methods = {None: cls._rvs_select(r, c, n), 'boyett': cls._rvs_boyett, 'patefield': cls._rvs_patefield}
        try:
            return known_methods[method]
        except KeyError:
            raise ValueError(f"'{method}' not recognized, must be one of {set(known_methods)}")

    @classmethod
    def _rvs_select(cls, r, c, n):
        fac = 1.0
        k = len(r) * len(c)
        if n > fac * np.log(n + 1) * k:
            return cls._rvs_patefield
        return cls._rvs_boyett

    @staticmethod
    def _rvs_boyett(row, col, ntot, size, random_state):
        return _rcont.rvs_rcont1(row, col, ntot, size, random_state)

    @staticmethod
    def _rvs_patefield(row, col, ntot, size, random_state):
        return _rcont.rvs_rcont2(row, col, ntot, size, random_state)