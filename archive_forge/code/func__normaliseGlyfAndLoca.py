from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def _normaliseGlyfAndLoca(self, padding=4):
    """Recompile glyf and loca tables, aligning glyph offsets to multiples of
        'padding' size. Update the head table's 'indexToLocFormat' accordingly while
        compiling loca.
        """
    if self.sfntVersion == 'OTTO':
        return
    for tag in ('maxp', 'head', 'loca', 'glyf', 'fvar'):
        if tag in self.tables:
            self._decompileTable(tag)
    self.ttFont['glyf'].padding = padding
    for tag in ('glyf', 'loca'):
        self._compileTable(tag)