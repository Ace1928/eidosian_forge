import numpy as np
from scipy.special import comb
def _convert_from_multidim(x, totype=list):
    if len(x.shape) < 2:
        return totype(x)
    return x.T