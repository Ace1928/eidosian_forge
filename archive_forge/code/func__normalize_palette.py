from __future__ import annotations
import itertools
import math
import os
import subprocess
from enum import IntEnum
from . import (
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16
def _normalize_palette(im, palette, info):
    """
    Normalizes the palette for image.
      - Sets the palette to the incoming palette, if provided.
      - Ensures that there's a palette for L mode images
      - Optimizes the palette if necessary/desired.

    :param im: Image object
    :param palette: bytes object containing the source palette, or ....
    :param info: encoderinfo
    :returns: Image object
    """
    source_palette = None
    if palette:
        if isinstance(palette, (bytes, bytearray, list)):
            source_palette = bytearray(palette[:768])
        if isinstance(palette, ImagePalette.ImagePalette):
            source_palette = bytearray(palette.palette)
    if im.mode == 'P':
        if not source_palette:
            source_palette = im.im.getpalette('RGB')[:768]
    else:
        if not source_palette:
            source_palette = bytearray((i // 3 for i in range(768)))
        im.palette = ImagePalette.ImagePalette('RGB', palette=source_palette)
    if palette:
        used_palette_colors = []
        for i in range(0, len(source_palette), 3):
            source_color = tuple(source_palette[i:i + 3])
            index = im.palette.colors.get(source_color)
            if index in used_palette_colors:
                index = None
            used_palette_colors.append(index)
        for i, index in enumerate(used_palette_colors):
            if index is None:
                for j in range(len(used_palette_colors)):
                    if j not in used_palette_colors:
                        used_palette_colors[i] = j
                        break
        im = im.remap_palette(used_palette_colors)
    else:
        used_palette_colors = _get_optimize(im, info)
        if used_palette_colors is not None:
            im = im.remap_palette(used_palette_colors, source_palette)
            if 'transparency' in info:
                try:
                    info['transparency'] = used_palette_colors.index(info['transparency'])
                except ValueError:
                    del info['transparency']
            return im
    im.palette.palette = source_palette
    return im