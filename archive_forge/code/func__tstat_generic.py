import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def _tstat_generic(value1, value2, std_diff, dof, alternative, diff=0):
    """generic ttest based on summary statistic

    The test statistic is :
        tstat = (value1 - value2 - diff) / std_diff

    and is assumed to be t-distributed with ``dof`` degrees of freedom.

    Parameters
    ----------
    value1 : float or ndarray
        Value, for example mean, of the first sample.
    value2 : float or ndarray
        Value, for example mean, of the second sample.
    std_diff : float or ndarray
        Standard error of the difference value1 - value2
    dof : int or float
        Degrees of freedom
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    diff : float
        value of difference ``value1 - value2`` under the null hypothesis

    Returns
    -------
    tstat : float or ndarray
        Test statistic.
    pvalue : float or ndarray
        P-value of the hypothesis test assuming that the test statistic is
        t-distributed with ``df`` degrees of freedom.
    """
    tstat = (value1 - value2 - diff) / std_diff
    if alternative in ['two-sided', '2-sided', '2s']:
        pvalue = stats.t.sf(np.abs(tstat), dof) * 2
    elif alternative in ['larger', 'l']:
        pvalue = stats.t.sf(tstat, dof)
    elif alternative in ['smaller', 's']:
        pvalue = stats.t.cdf(tstat, dof)
    else:
        raise ValueError('invalid alternative')
    return (tstat, pvalue)