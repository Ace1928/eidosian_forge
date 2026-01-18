import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
class MultiCompResult:
    """class to hold return of t_test_pairwise

    currently just a minimal class to hold attributes.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)