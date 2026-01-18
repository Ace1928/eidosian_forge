import numpy as np
from scipy.special import comb
def _convert_to_multidim(x):
    if any([isinstance(x, list), isinstance(x, tuple)]):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        return x