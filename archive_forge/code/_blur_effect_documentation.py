import numpy as np
import scipy.ndimage as ndi
from ..color import rgb2gray
from ..util import img_as_float
Compute a metric that indicates the strength of blur in an image
    (0 for no blur, 1 for maximal blur).

    Parameters
    ----------
    image : ndarray
        RGB or grayscale nD image. The input image is converted to grayscale
        before computing the blur metric.
    h_size : int, optional
        Size of the re-blurring filter.
    channel_axis : int or None, optional
        If None, the image is assumed to be grayscale (single-channel).
        Otherwise, this parameter indicates which axis of the array
        corresponds to color channels.
    reduce_func : callable, optional
        Function used to calculate the aggregation of blur metrics along all
        axes. If set to None, the entire list is returned, where the i-th
        element is the blur metric along the i-th axis.

    Returns
    -------
    blur : float (0 to 1) or list of floats
        Blur metric: by default, the maximum of blur metrics along all axes.

    Notes
    -----
    `h_size` must keep the same value in order to compare results between
    images. Most of the time, the default size (11) is enough. This means that
    the metric can clearly discriminate blur up to an average 11x11 filter; if
    blur is higher, the metric still gives good results but its values tend
    towards an asymptote.

    References
    ----------
    .. [1] Frederique Crete, Thierry Dolmiere, Patricia Ladret, and Marina
       Nicolas "The blur effect: perception and estimation with a new
       no-reference perceptual blur metric" Proc. SPIE 6492, Human Vision and
       Electronic Imaging XII, 64920I (2007)
       https://hal.archives-ouvertes.fr/hal-00232709
       :DOI:`10.1117/12.702790`
    