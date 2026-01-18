from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def Draw(im, mode=None):
    """
    A simple 2D drawing interface for PIL images.

    :param im: The image to draw in.
    :param mode: Optional mode to use for color values.  For RGB
       images, this argument can be RGB or RGBA (to blend the
       drawing into the image).  For all other modes, this argument
       must be the same as the image mode.  If omitted, the mode
       defaults to the mode of the image.
    """
    try:
        return im.getdraw(mode)
    except AttributeError:
        return ImageDraw(im, mode)