import random
import numpy as np
from ...block import Block, HybridBlock
from ...nn import Sequential, HybridSequential
from .... import image
from ....base import numeric_types
from ....util import is_np_array
class CropResize(HybridBlock):
    """Crop the input image with and optionally resize it.

    Makes a crop of the original image then optionally resize it to the specified size.

    Parameters
    ----------
    x : int
        Left boundary of the cropping area
    y : int
        Top boundary of the cropping area
    w : int
        Width of the cropping area
    h : int
        Height of the cropping area
    size : int or tuple of (w, h)
        Optional, resize to new size after cropping
    interpolation : int, optional
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=resize#resize
        Note that the Resize on gpu use contrib.bilinearResize2D operator
        which only support bilinear interpolation(1).


    Inputs:
        - **data**: input tensor with (H x W x C) or (N x H x W x C) shape.

    Outputs:
        - **out**: input tensor with (H x W x C) or (N x H x W x C) shape.

    Examples
    --------
    >>> transformer = vision.transforms.CropResize(x=0, y=0, width=100, height=100)
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 100x100x3 @cpu(0)>
    >>> image = mx.nd.random.uniform(0, 255, (3, 224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 3x100x100x3 @cpu(0)>
    >>> transformer = vision.transforms.CropResize(x=0, y=0, width=100, height=100, size=(50, 50), interpolation=1)
    >>> transformer(image)
    <NDArray 3x50x50 @cpu(0)>
    """

    def __init__(self, x, y, width, height, size=None, interpolation=None):
        super(CropResize, self).__init__()
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._size = size
        self._interpolation = interpolation

    def hybrid_forward(self, F, x):
        out = F.image.crop(x, self._x, self._y, self._width, self._height)
        if self._size:
            out = F.image.resize(out, self._size, False, self._interpolation)
        return out