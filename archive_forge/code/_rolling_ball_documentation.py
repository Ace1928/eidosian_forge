import numpy as np
from .._shared.utils import _supported_float_type
from ._rolling_ball_cy import apply_kernel, apply_kernel_nan
Create an ellipoid kernel for restoration.rolling_ball.

    Parameters
    ----------
    shape : array-like
        Length of the principal axis of the ellipsoid (excluding
        the intensity axis). The kernel needs to have the same
        dimensionality as the image it will be applied to.
    intensity : int
        Length of the intensity axis of the ellipsoid.

    Returns
    -------
    kernel : ndarray
        The kernel containing the surface intensity of the top half
        of the ellipsoid.

    See Also
    --------
    rolling_ball
    