import numpy as np
from scipy import stats
from statsmodels.compat.scipy import multivariate_t
from statsmodels.distributions.copula.copulas import Copula

        Bivariate tail dependence parameter.

        Joe (2014) p. 182

        Parameters
        ----------
        corr : None or float
            Pearson correlation. If corr is None, then the correlation will be
            taken from the copula attribute.

        Returns
        -------
        Lower and upper tail dependence coefficients of the copula with given
        Pearson correlation coefficient.
        