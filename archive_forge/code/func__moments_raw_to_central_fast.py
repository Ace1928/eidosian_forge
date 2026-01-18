import itertools
import math
import numpy as np
def _moments_raw_to_central_fast(moments_raw):
    """Analytical formulae for 2D and 3D central moments of order < 4.

    `moments_raw_to_central` will automatically call this function when
    ndim < 4 and order < 4.

    Parameters
    ----------
    moments_raw : ndarray
        The raw moments.

    Returns
    -------
    moments_central : ndarray
        The central moments.
    """
    ndim = moments_raw.ndim
    order = moments_raw.shape[0] - 1
    float_dtype = moments_raw.dtype
    moments_raw = moments_raw.astype(np.float64, copy=False)
    moments_central = np.zeros_like(moments_raw)
    if order >= 4 or ndim not in [2, 3]:
        raise ValueError('This function only supports 2D or 3D moments of order < 4.')
    m = moments_raw
    if ndim == 2:
        cx = m[1, 0] / m[0, 0]
        cy = m[0, 1] / m[0, 0]
        moments_central[0, 0] = m[0, 0]
        if order > 1:
            moments_central[1, 1] = m[1, 1] - cx * m[0, 1]
            moments_central[2, 0] = m[2, 0] - cx * m[1, 0]
            moments_central[0, 2] = m[0, 2] - cy * m[0, 1]
        if order > 2:
            moments_central[2, 1] = m[2, 1] - 2 * cx * m[1, 1] - cy * m[2, 0] + cx ** 2 * m[0, 1] + cy * cx * m[1, 0]
            moments_central[1, 2] = m[1, 2] - 2 * cy * m[1, 1] - cx * m[0, 2] + 2 * cy * cx * m[0, 1]
            moments_central[3, 0] = m[3, 0] - 3 * cx * m[2, 0] + 2 * cx ** 2 * m[1, 0]
            moments_central[0, 3] = m[0, 3] - 3 * cy * m[0, 2] + 2 * cy ** 2 * m[0, 1]
    else:
        cx = m[1, 0, 0] / m[0, 0, 0]
        cy = m[0, 1, 0] / m[0, 0, 0]
        cz = m[0, 0, 1] / m[0, 0, 0]
        moments_central[0, 0, 0] = m[0, 0, 0]
        if order > 1:
            moments_central[0, 0, 2] = -cz * m[0, 0, 1] + m[0, 0, 2]
            moments_central[0, 1, 1] = -cy * m[0, 0, 1] + m[0, 1, 1]
            moments_central[0, 2, 0] = -cy * m[0, 1, 0] + m[0, 2, 0]
            moments_central[1, 0, 1] = -cx * m[0, 0, 1] + m[1, 0, 1]
            moments_central[1, 1, 0] = -cx * m[0, 1, 0] + m[1, 1, 0]
            moments_central[2, 0, 0] = -cx * m[1, 0, 0] + m[2, 0, 0]
        if order > 2:
            moments_central[0, 0, 3] = 2 * cz ** 2 * m[0, 0, 1] - 3 * cz * m[0, 0, 2] + m[0, 0, 3]
            moments_central[0, 1, 2] = -cy * m[0, 0, 2] + 2 * cz * (cy * m[0, 0, 1] - m[0, 1, 1]) + m[0, 1, 2]
            moments_central[0, 2, 1] = cy ** 2 * m[0, 0, 1] - 2 * cy * m[0, 1, 1] + cz * (cy * m[0, 1, 0] - m[0, 2, 0]) + m[0, 2, 1]
            moments_central[0, 3, 0] = 2 * cy ** 2 * m[0, 1, 0] - 3 * cy * m[0, 2, 0] + m[0, 3, 0]
            moments_central[1, 0, 2] = -cx * m[0, 0, 2] + 2 * cz * (cx * m[0, 0, 1] - m[1, 0, 1]) + m[1, 0, 2]
            moments_central[1, 1, 1] = -cx * m[0, 1, 1] + cy * (cx * m[0, 0, 1] - m[1, 0, 1]) + cz * (cx * m[0, 1, 0] - m[1, 1, 0]) + m[1, 1, 1]
            moments_central[1, 2, 0] = -cx * m[0, 2, 0] - 2 * cy * (-cx * m[0, 1, 0] + m[1, 1, 0]) + m[1, 2, 0]
            moments_central[2, 0, 1] = cx ** 2 * m[0, 0, 1] - 2 * cx * m[1, 0, 1] + cz * (cx * m[1, 0, 0] - m[2, 0, 0]) + m[2, 0, 1]
            moments_central[2, 1, 0] = cx ** 2 * m[0, 1, 0] - 2 * cx * m[1, 1, 0] + cy * (cx * m[1, 0, 0] - m[2, 0, 0]) + m[2, 1, 0]
            moments_central[3, 0, 0] = 2 * cx ** 2 * m[1, 0, 0] - 3 * cx * m[2, 0, 0] + m[3, 0, 0]
    return moments_central.astype(float_dtype, copy=False)