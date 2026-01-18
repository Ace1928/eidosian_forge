from __future__ import annotations
from . import ImageFile, ImagePalette, UnidentifiedImageError
from ._binary import i16be as i16
from ._binary import i32be as i32

    Load texture from a GD image file.

    :param fp: GD file name, or an opened file handle.
    :param mode: Optional mode.  In this version, if the mode argument
        is given, it must be "r".
    :returns: An image instance.
    :raises OSError: If the image could not be read.
    