import warnings
import numpy as np
from scipy.special import expm1, gamma
def _checkargs(self, theta):
    return theta >= 1