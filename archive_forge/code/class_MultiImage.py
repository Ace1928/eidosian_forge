import os
from glob import glob
import re
from collections.abc import Sequence
from copy import copy
import numpy as np
from PIL import Image
from tifffile import TiffFile
class MultiImage(ImageCollection):
    """A class containing all frames from multi-frame TIFF images.

    Parameters
    ----------
    load_pattern : str or list of str
        Pattern glob or filenames to load. The path can be absolute or
        relative.
    conserve_memory : bool, optional
        Whether to conserve memory by only caching the frames of a single
        image. Default is True.

    Notes
    -----
    `MultiImage` returns a list of image-data arrays. In this
    regard, it is very similar to `ImageCollection`, but the two differ in
    their treatment of multi-frame images.

    For a TIFF image containing N frames of size WxH, `MultiImage` stores
    all frames of that image as a single element of shape `(N, W, H)` in the
    list. `ImageCollection` instead creates N elements of shape `(W, H)`.

    For an animated GIF image, `MultiImage` reads only the first frame, while
    `ImageCollection` reads all frames by default.

    Examples
    --------
    # Where your images are located
    >>> data_dir = os.path.join(os.path.dirname(__file__), '../data')

    >>> multipage_tiff = data_dir + '/multipage.tif'
    >>> multi_img = MultiImage(multipage_tiff)
    >>> len(multi_img)  # multi_img contains one element
    1
    >>> multi_img[0].shape  # this element is a two-frame image of shape:
    (2, 15, 10)

    >>> image_col = ImageCollection(multipage_tiff)
    >>> len(image_col)  # image_col contains two elements
    2
    >>> for frame in image_col:
    ...     print(frame.shape)  # each element is a frame of shape (15, 10)
    ...
    (15, 10)
    (15, 10)
    """

    def __init__(self, filename, conserve_memory=True, dtype=None, **imread_kwargs):
        """Load a multi-img."""
        from ._io import imread
        self._filename = filename
        super().__init__(filename, conserve_memory, load_func=imread, **imread_kwargs)

    @property
    def filename(self):
        return self._filename