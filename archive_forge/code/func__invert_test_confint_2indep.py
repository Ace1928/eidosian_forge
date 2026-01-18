import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def _invert_test_confint_2indep(count1, exposure1, count2, exposure2, alpha=0.05, method='score', compare='diff', method_start='wald'):
    """invert hypothesis test to get confidence interval for 2indep
    """

    def func(r):
        v = (test_poisson_2indep(count1, exposure1, count2, exposure2, value=r, method=method, compare=compare)[1] - alpha) ** 2
        return v
    ci = confint_poisson_2indep(count1, exposure1, count2, exposure2, method=method_start, compare=compare)
    low = optimize.fmin(func, ci[0], xtol=1e-08, disp=False)
    upp = optimize.fmin(func, ci[1], xtol=1e-08, disp=False)
    assert np.size(low) == 1
    return (low[0], upp[0])