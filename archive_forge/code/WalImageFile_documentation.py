from __future__ import annotations
from . import Image, ImageFile
from ._binary import i32le as i32

    Load texture from a Quake2 WAL texture file.

    By default, a Quake2 standard palette is attached to the texture.
    To override the palette, use the :py:func:`PIL.Image.Image.putpalette()` method.

    :param filename: WAL file name, or an opened file handle.
    :returns: An image instance.
    