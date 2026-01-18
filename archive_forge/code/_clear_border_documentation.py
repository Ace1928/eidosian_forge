import numpy as np
from ..measure import label
Clear objects connected to the label image border.

    Parameters
    ----------
    labels : (M[, N[, ..., P]]) array of int or bool
        Imaging data labels.
    buffer_size : int, optional
        The width of the border examined.  By default, only objects
        that touch the outside of the image are removed.
    bgval : float or int, optional
        Cleared objects are set to this value.
    mask : ndarray of bool, same shape as `image`, optional.
        Image data mask. Objects in labels image overlapping with
        False pixels of mask will be removed. If defined, the
        argument buffer_size will be ignored.
    out : ndarray
        Array of the same shape as `labels`, into which the
        output is placed. By default, a new array is created.

    Returns
    -------
    out : (M[, N[, ..., P]]) array
        Imaging data labels with cleared borders

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.segmentation import clear_border
    >>> labels = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0],
    ...                    [1, 1, 0, 0, 1, 0, 0, 1, 0],
    ...                    [1, 1, 0, 1, 0, 1, 0, 0, 0],
    ...                    [0, 0, 0, 1, 1, 1, 1, 0, 0],
    ...                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> clear_border(labels)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> mask = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1],
    ...                  [0, 0, 1, 1, 1, 1, 1, 1, 1],
    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)
    >>> clear_border(labels, mask=mask)
    array([[0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0, 1, 0],
           [0, 0, 0, 1, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    