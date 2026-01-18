from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
def a_st70_upp(stat, nobs):
    nobsinv = 1.0 / nobs
    stat_modified = stat - 0.7 * nobsinv + 0.9 * nobsinv ** 2
    stat_modified *= 1 + 1.23 * nobsinv
    pval = 1.273 * np.exp(-2 * stat_modified / 2.0 * np.pi ** 2)
    digits = np.sum(stat > np.array([0.11, 0.11, 0.452]))
    return (stat_modified, pval, digits)