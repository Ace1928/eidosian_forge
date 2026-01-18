import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
def _tconfint_generic(mean, std_mean, dof, alpha, alternative):
    """generic t-confint based on summary statistic

    Parameters
    ----------
    mean : float or ndarray
        Value, for example mean, of the first sample.
    std_mean : float or ndarray
        Standard error of the difference value1 - value2
    dof : int or float
        Degrees of freedom
    alpha : float
        Significance level for the confidence interval, coverage is
        ``1-alpha``.
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
        tcrit = stats.t.ppf(1 - alpha / 2.0, dof)
        lower = mean - tcrit * std_mean
        upper = mean + tcrit * std_mean
    elif alternative in ['larger', 'l']:
        tcrit = stats.t.ppf(alpha, dof)
        lower = mean + tcrit * std_mean
        upper = np.inf
    elif alternative in ['smaller', 's']:
        tcrit = stats.t.ppf(1 - alpha, dof)
        lower = -np.inf
        upper = mean + tcrit * std_mean
    else:
        raise ValueError('invalid alternative')
    return (lower, upper)