from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def _ptransform(p):
    """function for p-value abcissa transformation"""
    return -1.0 / (1.0 + 1.5 * _phi((1.0 + p) / 2.0))