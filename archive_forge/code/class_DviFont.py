from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import numpy as np
from matplotlib import _api, cbook
class DviFont:
    """
    Encapsulation of a font that a DVI file can refer to.

    This class holds a font's texname and size, supports comparison,
    and knows the widths of glyphs in the same units as the AFM file.
    There are also internal attributes (for use by dviread.py) that
    are *not* used for comparison.

    The size is in Adobe points (converted from TeX points).

    Parameters
    ----------
    scale : float
        Factor by which the font is scaled from its natural size.
    tfm : Tfm
        TeX font metrics for this font
    texname : bytes
       Name of the font as used internally by TeX and friends, as an ASCII
       bytestring.  This is usually very different from any external font
       names; `PsfontsMap` can be used to find the external name of the font.
    vf : Vf
       A TeX "virtual font" file, or None if this font is not virtual.

    Attributes
    ----------
    texname : bytes
    size : float
       Size of the font in Adobe points, converted from the slightly
       smaller TeX points.
    widths : list
       Widths of glyphs in glyph-space units, typically 1/1000ths of
       the point size.

    """
    __slots__ = ('texname', 'size', 'widths', '_scale', '_vf', '_tfm')

    def __init__(self, scale, tfm, texname, vf):
        _api.check_isinstance(bytes, texname=texname)
        self._scale = scale
        self._tfm = tfm
        self.texname = texname
        self._vf = vf
        self.size = scale * (72.0 / (72.27 * 2 ** 16))
        try:
            nchars = max(tfm.width) + 1
        except ValueError:
            nchars = 0
        self.widths = [1000 * tfm.width.get(char, 0) >> 20 for char in range(nchars)]

    def __eq__(self, other):
        return type(self) is type(other) and self.texname == other.texname and (self.size == other.size)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f'<{type(self).__name__}: {self.texname}>'

    def _width_of(self, char):
        """Width of char in dvi units."""
        width = self._tfm.width.get(char, None)
        if width is not None:
            return _mul2012(width, self._scale)
        _log.debug('No width for char %d in font %s.', char, self.texname)
        return 0

    def _height_depth_of(self, char):
        """Height and depth of char in dvi units."""
        result = []
        for metric, name in ((self._tfm.height, 'height'), (self._tfm.depth, 'depth')):
            value = metric.get(char, None)
            if value is None:
                _log.debug('No %s for char %d in font %s', name, char, self.texname)
                result.append(0)
            else:
                result.append(_mul2012(value, self._scale))
        if re.match(b'^cmsy\\d+$', self.texname) and char == 0:
            result[-1] = 0
        return result