from __future__ import annotations
import re
from . import Image, _imagingmorph
class MorphOp:
    """A class for binary morphological operators"""

    def __init__(self, lut=None, op_name=None, patterns=None):
        """Create a binary morphological operator"""
        self.lut = lut
        if op_name is not None:
            self.lut = LutBuilder(op_name=op_name).build_lut()
        elif patterns is not None:
            self.lut = LutBuilder(patterns=patterns).build_lut()

    def apply(self, image):
        """Run a single morphological operation on an image

        Returns a tuple of the number of changed pixels and the
        morphed image"""
        if self.lut is None:
            msg = 'No operator loaded'
            raise Exception(msg)
        if image.mode != 'L':
            msg = 'Image mode must be L'
            raise ValueError(msg)
        outimage = Image.new(image.mode, image.size, None)
        count = _imagingmorph.apply(bytes(self.lut), image.im.id, outimage.im.id)
        return (count, outimage)

    def match(self, image):
        """Get a list of coordinates matching the morphological operation on
        an image.

        Returns a list of tuples of (x,y) coordinates
        of all matching pixels. See :ref:`coordinate-system`."""
        if self.lut is None:
            msg = 'No operator loaded'
            raise Exception(msg)
        if image.mode != 'L':
            msg = 'Image mode must be L'
            raise ValueError(msg)
        return _imagingmorph.match(bytes(self.lut), image.im.id)

    def get_on_pixels(self, image):
        """Get a list of all turned on pixels in a binary image

        Returns a list of tuples of (x,y) coordinates
        of all matching pixels. See :ref:`coordinate-system`."""
        if image.mode != 'L':
            msg = 'Image mode must be L'
            raise ValueError(msg)
        return _imagingmorph.get_on_pixels(image.im.id)

    def load_lut(self, filename):
        """Load an operator from an mrl file"""
        with open(filename, 'rb') as f:
            self.lut = bytearray(f.read())
        if len(self.lut) != LUT_SIZE:
            self.lut = None
            msg = 'Wrong size operator file!'
            raise Exception(msg)

    def save_lut(self, filename):
        """Save an operator to an mrl file"""
        if self.lut is None:
            msg = 'No operator loaded'
            raise Exception(msg)
        with open(filename, 'wb') as f:
            f.write(self.lut)

    def set_lut(self, lut):
        """Set the lut from an external source"""
        self.lut = lut