import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _fast_kde_2d(x, y, gridsize=(128, 128), circular=False):
    """
    2D fft-based Gaussian kernel density estimate (KDE).

    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------
    x : Numpy array or list
    y : Numpy array or list
    gridsize : tuple
        Number of points used to discretize data. Use powers of 2 for fft optimization
    circular: bool
        If True use circular boundaries. Defaults to False

    Returns
    -------
    grid: A gridded 2D KDE of the input points (x, y)
    xmin: minimum value of x
    xmax: maximum value of x
    ymin: minimum value of y
    ymax: maximum value of y
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    xmin, xmax = (x.min(), x.max())
    ymin, ymax = (y.min(), y.max())
    len_x = len(x)
    weights = np.ones(len_x)
    n_x, n_y = gridsize
    d_x = (xmax - xmin) / (n_x - 1)
    d_y = (ymax - ymin) / (n_y - 1)
    xyi = _stack(x, y).T
    xyi -= [xmin, ymin]
    xyi /= [d_x, d_y]
    xyi = np.floor(xyi, xyi).T
    scotts_factor = len_x ** (-1 / 6)
    cov = _cov(xyi)
    std_devs = np.diag(cov) ** 0.5
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)
    inv_cov = np.linalg.inv(cov * scotts_factor ** 2)
    x_x = np.arange(kern_nx) - kern_nx / 2
    y_y = np.arange(kern_ny) - kern_ny / 2
    x_x, y_y = np.meshgrid(x_x, y_y)
    kernel = _stack(x_x.flatten(), y_y.flatten())
    kernel = _dot(inv_cov, kernel) * kernel
    kernel = np.exp(-kernel.sum(axis=0) / 2)
    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))
    boundary = 'wrap' if circular else 'symm'
    grid = coo_matrix((weights, xyi), shape=(n_x, n_y)).toarray()
    grid = convolve2d(grid, kernel, mode='same', boundary=boundary)
    norm_factor = np.linalg.det(2 * np.pi * cov * scotts_factor ** 2)
    norm_factor = len_x * d_x * d_y * norm_factor ** 0.5
    grid /= norm_factor
    return (grid, xmin, xmax, ymin, ymax)