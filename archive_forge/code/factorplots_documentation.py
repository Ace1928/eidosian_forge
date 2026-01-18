from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.graphics.plottools import rainbow
import statsmodels.graphics.utils as utils
 Recode categorial data to int factor.

    Parameters
    ----------
    x : array_like
        array like object supporting with numpy array methods of categorially
        coded data.
    levels : dict
        mapping of labels to integer-codings

    Returns
    -------
    out : instance numpy.ndarray
    