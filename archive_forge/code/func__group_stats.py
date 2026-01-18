import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
def _group_stats(self, groups):
    """
        Descriptive statistics of the groups.
        """
    gsizes = np.unique(groups, return_counts=True)
    gsizes = gsizes[1]
    return (gsizes.min(), gsizes.max(), gsizes.mean(), len(gsizes))