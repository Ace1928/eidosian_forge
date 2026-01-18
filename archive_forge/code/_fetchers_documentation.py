from numpy import array, frombuffer, load
from ._registry import registry, registry_urls

    Get a 1024 x 768, color image of a raccoon face.

    raccoon-procyon-lotor.jpg at http://www.public-domain-image.com

    Parameters
    ----------
    gray : bool, optional
        If True return 8-bit grey-scale image, otherwise return a color image

    Returns
    -------
    face : ndarray
        image of a raccoon face

    Examples
    --------
    >>> import scipy.datasets
    >>> face = scipy.datasets.face()
    >>> face.shape
    (768, 1024, 3)
    >>> face.max()
    255
    >>> face.dtype
    dtype('uint8')

    >>> import matplotlib.pyplot as plt
    >>> plt.gray()
    >>> plt.imshow(face)
    >>> plt.show()

    