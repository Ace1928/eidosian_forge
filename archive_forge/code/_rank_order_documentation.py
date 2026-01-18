import numpy as np
Return an image of the same shape where each pixel is the
    index of the pixel value in the ascending order of the unique
    values of ``image``, aka the rank-order value.

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    labels : ndarray of unsigned integers, of shape image.shape
        New array where each pixel has the rank-order value of the
        corresponding pixel in ``image``. Pixel values are between 0 and
        n - 1, where n is the number of distinct unique values in
        ``image``. The dtype of this array will be determined by
        ``np.min_scalar_type(image.size)``.
    original_values : 1-D ndarray
        Unique original values of ``image``. This will have the same dtype as
        ``image``.

    Examples
    --------
    >>> a = np.array([[1, 4, 5], [4, 4, 1], [5, 1, 1]])
    >>> a
    array([[1, 4, 5],
           [4, 4, 1],
           [5, 1, 1]])
    >>> rank_order(a)
    (array([[0, 1, 2],
           [1, 1, 0],
           [2, 0, 0]], dtype=uint8), array([1, 4, 5]))
    >>> b = np.array([-1., 2.5, 3.1, 2.5])
    >>> rank_order(b)
    (array([0, 1, 2, 1], dtype=uint8), array([-1. ,  2.5,  3.1]))
    