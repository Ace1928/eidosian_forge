from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.graphics.plottools import rainbow
import statsmodels.graphics.utils as utils
def _recode(x, levels):
    """ Recode categorial data to int factor.

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
    """
    from pandas import Series
    name = None
    index = None
    if isinstance(x, Series):
        name = x.name
        index = x.index
        x = x.values
    if x.dtype.type not in [np.str_, np.object_]:
        raise ValueError('This is not a categorial factor. Array of str type required.')
    elif not isinstance(levels, dict):
        raise ValueError('This is not a valid value for levels. Dict required.')
    elif not (np.unique(x) == np.unique(list(levels.keys()))).all():
        raise ValueError('The levels do not match the array values.')
    else:
        out = np.empty(x.shape[0], dtype=int)
        for level, coding in levels.items():
            out[x == level] = coding
        if name:
            out = Series(out, name=name, index=index)
        return out