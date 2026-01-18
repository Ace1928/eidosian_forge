import os
import numpy as np
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test

    Prints descriptive statistics for one or multiple variables.

    Parameters
    ----------
    data: numpy array
        `x` is the data

    v: list, optional
        A list of the column number of variables.
        Default is all columns.

    axis: 1 or 0
        axis order of data.  Default is 0 for column-ordered data.

    Examples
    --------
    >>> descstats(data.exog,v=['x_1','x_2','x_3'])
    