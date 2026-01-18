import warnings
import numpy as np
from scipy.linalg import eig
from scipy.special import comb
from scipy.signal import convolve
def _ricker(points, a):
    A = 2 / (np.sqrt(3 * a) * np.pi ** 0.25)
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec ** 2
    mod = 1 - xsq / wsq
    gauss = np.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total