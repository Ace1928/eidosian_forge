import numpy as np
from scipy.interpolate import interp1d
class ECDFDiscrete(StepFunction):
    """
    Return the Empirical Weighted CDF of an array as a step function.

    Parameters
    ----------
    x : array_like
        Data values. If freq_weights is None, then x is treated as observations
        and the ecdf is computed from the frequency counts of unique values
        using nunpy.unique.
        If freq_weights is not None, then x will be taken as the support of the
        mass point distribution with freq_weights as counts for x values.
        The x values can be arbitrary sortable values and need not be integers.
    freq_weights : array_like
        Weights of the observations.  sum(freq_weights) is interpreted as nobs
        for confint.
        If freq_weights is None, then the frequency counts for unique values
        will be computed from the data x.
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Weighted ECDF as a step function.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import (
    >>>     ECDFDiscrete)
    >>>
    >>> ewcdf = ECDFDiscrete([3, 3, 1, 4])
    >>> ewcdf([3, 55, 0.5, 1.5])
    array([0.75, 1.  , 0.  , 0.25])
    >>>
    >>> ewcdf = ECDFDiscrete([3, 1, 4], [1.25, 2.5, 5])
    >>>
    >>> ewcdf([3, 55, 0.5, 1.5])
    array([0.42857143, 1., 0. , 0.28571429])
    >>> print('e1 and e2 are equivalent ways of defining the same ECDF')
    e1 and e2 are equivalent ways of defining the same ECDF
    >>> e1 = ECDFDiscrete([3.5, 3.5, 1.5, 1, 4])
    >>> e2 = ECDFDiscrete([3.5, 1.5, 1, 4], freq_weights=[2, 1, 1, 1])
    >>> print(e1.x, e2.x)
    [-inf  1.   1.5  3.5  4. ] [-inf  1.   1.5  3.5  4. ]
    >>> print(e1.y, e2.y)
    [0.  0.2 0.4 0.8 1. ] [0.  0.2 0.4 0.8 1. ]
    """

    def __init__(self, x, freq_weights=None, side='right'):
        if freq_weights is None:
            x, freq_weights = np.unique(x, return_counts=True)
        else:
            x = np.asarray(x)
        assert len(freq_weights) == len(x)
        w = np.asarray(freq_weights)
        sw = np.sum(w)
        assert sw > 0
        ax = x.argsort()
        x = x[ax]
        y = np.cumsum(w[ax])
        y = y / sw
        super().__init__(x, y, side=side, sorted=True)