from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate

        Returns the pixel at x,y. The pixel is returned as a single
        value for single band images or a tuple for multiple band
        images

        :param xy: The pixel coordinate, given as (x, y). See
          :ref:`coordinate-system`.
        :returns: a pixel value for single band images, a tuple of
          pixel values for multiband images.
        