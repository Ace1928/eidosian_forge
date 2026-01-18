from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def _select_ps(p):
    """returns the points to use for interpolating p"""
    if p >= 0.99:
        return (0.99, 0.995, 0.999)
    elif p >= 0.975:
        return (0.975, 0.99, 0.995)
    elif p >= 0.95:
        return (0.95, 0.975, 0.99)
    elif p >= 0.9125:
        return (0.9, 0.95, 0.975)
    elif p >= 0.875:
        return (0.85, 0.9, 0.95)
    elif p >= 0.825:
        return (0.8, 0.85, 0.9)
    elif p >= 0.7625:
        return (0.75, 0.8, 0.85)
    elif p >= 0.675:
        return (0.675, 0.75, 0.8)
    elif p >= 0.5:
        return (0.5, 0.675, 0.75)
    else:
        return (0.1, 0.5, 0.675)