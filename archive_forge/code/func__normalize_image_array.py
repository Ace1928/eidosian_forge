import math
import os
import logging
from pathlib import Path
import warnings
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import matplotlib as mpl
from matplotlib import _api, cbook, cm
from matplotlib import _image
from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
@staticmethod
def _normalize_image_array(A):
    """
        Check validity of image-like input *A* and normalize it to a format suitable for
        Image subclasses.
        """
    A = cbook.safe_masked_invalid(A, copy=True)
    if A.dtype != np.uint8 and (not np.can_cast(A.dtype, float, 'same_kind')):
        raise TypeError(f'Image data of dtype {A.dtype} cannot be converted to float')
    if A.ndim == 3 and A.shape[-1] == 1:
        A = A.squeeze(-1)
    if not (A.ndim == 2 or (A.ndim == 3 and A.shape[-1] in [3, 4])):
        raise TypeError(f'Invalid shape {A.shape} for image data')
    if A.ndim == 3:
        high = 255 if np.issubdtype(A.dtype, np.integer) else 1
        if A.min() < 0 or high < A.max():
            _log.warning('Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).')
            A = np.clip(A, 0, high)
        if A.dtype != np.uint8 and np.issubdtype(A.dtype, np.integer):
            A = A.astype(np.uint8)
    return A