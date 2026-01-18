import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def _zconfint_generic(mean, std_mean, alpha, alternative):
    """generic normal-confint based on summary statistic

    Parameters
    ----------
    mean : float or ndarray
        Value, for example mean, of the first sample.
    std_mean : float or ndarray
        Standard error of the difference value1 - value2
    alpha : float
        Significance level for the confidence interval, coverage is
        ``1-alpha``
    alternative : str
        The alternative hypothesis, H1, has to be one of the following

           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.
           * 'larger' :   H1: ``value1 - value2 - diff > 0``
           * 'smaller' :  H1: ``value1 - value2 - diff < 0``

    Returns
    -------
    lower : float or ndarray
        Lower confidence limit. This is -inf for the one-sided alternative
        "smaller".
    upper : float or ndarray
        Upper confidence limit. This is inf for the one-sided alternative
        "larger".
    """
    if alternative in ['two-sided', '2-sided', '2s']:
        zcrit = stats.norm.ppf(1 - alpha / 2.0)
        lower = mean - zcrit * std_mean
        upper = mean + zcrit * std_mean
    elif alternative in ['larger', 'l']:
        zcrit = stats.norm.ppf(alpha)
        lower = mean + zcrit * std_mean
        upper = np.inf
    elif alternative in ['smaller', 's']:
        zcrit = stats.norm.ppf(1 - alpha)
        lower = -np.inf
        upper = mean + zcrit * std_mean
    else:
        raise ValueError('invalid alternative')
    return (lower, upper)