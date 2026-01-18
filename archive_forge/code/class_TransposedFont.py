from __future__ import annotations
import base64
import os
import sys
import warnings
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from . import Image
from ._util import is_directory, is_path
class TransposedFont:
    """Wrapper for writing rotated or mirrored text"""

    def __init__(self, font, orientation=None):
        """
        Wrapper that creates a transposed font from any existing font
        object.

        :param font: A font object.
        :param orientation: An optional orientation.  If given, this should
            be one of Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, or
            Image.Transpose.ROTATE_270.
        """
        self.font = font
        self.orientation = orientation

    def getmask(self, text, mode='', *args, **kwargs):
        im = self.font.getmask(text, mode, *args, **kwargs)
        if self.orientation is not None:
            return im.transpose(self.orientation)
        return im

    def getbbox(self, text, *args, **kwargs):
        left, top, right, bottom = self.font.getbbox(text, *args, **kwargs)
        width = right - left
        height = bottom - top
        if self.orientation in (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270):
            return (0, 0, height, width)
        return (0, 0, width, height)

    def getlength(self, text, *args, **kwargs):
        if self.orientation in (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270):
            msg = 'text length is undefined for text rotated by 90 or 270 degrees'
            raise ValueError(msg)
        return self.font.getlength(text, *args, **kwargs)