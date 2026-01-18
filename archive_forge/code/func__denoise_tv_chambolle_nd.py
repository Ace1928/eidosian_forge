import functools
from math import ceil
import numbers
import scipy.stats
import numpy as np
from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.utils import _supported_float_type, warn
from ._denoise_cy import _denoise_bilateral, _denoise_tv_bregman
from .. import color
from ..color.colorconv import ycbcr_from_rgb
def _denoise_tv_chambolle_nd(image, weight=0.1, eps=0.0002, max_num_iter=200):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    image : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    max_num_iter : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """
    ndim = image.ndim
    p = np.zeros((image.ndim,) + image.shape, dtype=image.dtype)
    g = np.zeros_like(p)
    d = np.zeros_like(image)
    i = 0
    while i < max_num_iter:
        if i > 0:
            d = -p.sum(0)
            slices_d = [slice(None)] * ndim
            slices_p = [slice(None)] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax + 1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax + 1] = slice(None)
            out = image + d
        else:
            out = image
        E = (d ** 2).sum()
        slices_g = [slice(None)] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax + 1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = np.diff(out, axis=ax)
            slices_g[ax + 1] = slice(None)
        norm = np.sqrt((g ** 2).sum(axis=0))[np.newaxis, ...]
        E += weight * norm.sum()
        tau = 1.0 / (2.0 * ndim)
        norm *= tau / weight
        norm += 1.0
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        elif np.abs(E_previous - E) < eps * E_init:
            break
        else:
            E_previous = E
        i += 1
    return out