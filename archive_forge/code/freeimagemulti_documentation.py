import logging
import numpy as np
from ..core import Format, image_as_uint
from ._freeimage import fi, IO_FLAGS
from .freeimage import FreeimageFormat

            Calculate the minimal rectangles that need updating each frame.
            Returns a two-element tuple containing the cropped images and a
            list of x-y positions.
            