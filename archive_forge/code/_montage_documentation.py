import numpy as np
from .._shared import utils
from .. import exposure
Create a montage of several single- or multichannel images.

    Create a rectangular montage from an input array representing an ensemble
    of equally shaped single- (gray) or multichannel (color) images.

    For example, ``montage(arr_in)`` called with the following `arr_in`

    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+

    will return

    +---+---+
    | 1 | 2 |
    +---+---+
    | 3 | * |
    +---+---+

    where the '*' patch will be determined by the `fill` parameter.

    Parameters
    ----------
    arr_in : ndarray, shape (K, M, N[, C])
        An array representing an ensemble of `K` images of equal shape.
    fill : float or array-like of floats or 'mean', optional
        Value to fill the padding areas and/or the extra tiles in
        the output array. Has to be `float` for single channel collections.
        For multichannel collections has to be an array-like of shape of
        number of channels. If `mean`, uses the mean value over all images.
    rescale_intensity : bool, optional
        Whether to rescale the intensity of each image to [0, 1].
    grid_shape : tuple, optional
        The desired grid shape for the montage `(ntiles_row, ntiles_column)`.
        The default aspect ratio is square.
    padding_width : int, optional
        The size of the spacing between the tiles and between the tiles and
        the borders. If non-zero, makes the boundaries of individual images
        easier to perceive.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    arr_out : (K*(M+p)+p, K*(N+p)+p[, C]) ndarray
        Output array with input images glued together (including padding `p`).

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util import montage
    >>> arr_in = np.arange(3 * 2 * 2).reshape(3, 2, 2)
    >>> arr_in  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1],
            [ 2,  3]],
           [[ 4,  5],
            [ 6,  7]],
           [[ 8,  9],
            [10, 11]]])
    >>> arr_out = montage(arr_in)
    >>> arr_out.shape
    (4, 4)
    >>> arr_out
    array([[ 0,  1,  4,  5],
           [ 2,  3,  6,  7],
           [ 8,  9,  5,  5],
           [10, 11,  5,  5]])
    >>> arr_in.mean()
    5.5
    >>> arr_out_nonsquare = montage(arr_in, grid_shape=(1, 3))
    >>> arr_out_nonsquare
    array([[ 0,  1,  4,  5,  8,  9],
           [ 2,  3,  6,  7, 10, 11]])
    >>> arr_out_nonsquare.shape
    (2, 6)
    