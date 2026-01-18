import math
import numpy as np
from numpy import arctan2, exp, pi, sqrt
from .. import draw
from ..util.dtype import img_as_float
from .._shared.filters import gaussian
from .._shared.utils import check_nD
from ..color import gray2rgb
Extract DAISY feature descriptors densely for the given image.

    DAISY is a feature descriptor similar to SIFT formulated in a way that
    allows for fast dense extraction. Typically, this is practical for
    bag-of-features image representations.

    The implementation follows Tola et al. [1]_ but deviate on the following
    points:

      * Histogram bin contribution are smoothed with a circular Gaussian
        window over the tonal range (the angular range).
      * The sigma values of the spatial Gaussian smoothing in this code do not
        match the sigma values in the original code by Tola et al. [2]_. In
        their code, spatial smoothing is applied to both the input image and
        the center histogram. However, this smoothing is not documented in [1]_
        and, therefore, it is omitted.

    Parameters
    ----------
    image : (M, N) array
        Input image (grayscale).
    step : int, optional
        Distance between descriptor sampling points.
    radius : int, optional
        Radius (in pixels) of the outermost ring.
    rings : int, optional
        Number of rings.
    histograms : int, optional
        Number of histograms sampled per ring.
    orientations : int, optional
        Number of orientations (bins) per histogram.
    normalization : [ 'l1' | 'l2' | 'daisy' | 'off' ], optional
        How to normalize the descriptors

          * 'l1': L1-normalization of each descriptor.
          * 'l2': L2-normalization of each descriptor.
          * 'daisy': L2-normalization of individual histograms.
          * 'off': Disable normalization.

    sigmas : 1D array of float, optional
        Standard deviation of spatial Gaussian smoothing for the center
        histogram and for each ring of histograms. The array of sigmas should
        be sorted from the center and out. I.e. the first sigma value defines
        the spatial smoothing of the center histogram and the last sigma value
        defines the spatial smoothing of the outermost ring. Specifying sigmas
        overrides the following parameter.

            ``rings = len(sigmas) - 1``

    ring_radii : 1D array of int, optional
        Radius (in pixels) for each ring. Specifying ring_radii overrides the
        following two parameters.

            ``rings = len(ring_radii)``
            ``radius = ring_radii[-1]``

        If both sigmas and ring_radii are given, they must satisfy the
        following predicate since no radius is needed for the center
        histogram.

            ``len(ring_radii) == len(sigmas) + 1``

    visualize : bool, optional
        Generate a visualization of the DAISY descriptors

    Returns
    -------
    descs : array
        Grid of DAISY descriptors for the given image as an array
        dimensionality  (P, Q, R) where

            ``P = ceil((M - radius*2) / step)``
            ``Q = ceil((N - radius*2) / step)``
            ``R = (rings * histograms + 1) * orientations``

    descs_img : (M, N, 3) array (only if visualize==True)
        Visualization of the DAISY descriptors.

    References
    ----------
    .. [1] Tola et al. "Daisy: An efficient dense descriptor applied to wide-
           baseline stereo." Pattern Analysis and Machine Intelligence, IEEE
           Transactions on 32.5 (2010): 815-830.
    .. [2] http://cvlab.epfl.ch/software/daisy
    