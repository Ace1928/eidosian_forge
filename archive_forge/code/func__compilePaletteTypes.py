from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import array
from collections import namedtuple
import struct
import sys
def _compilePaletteTypes(self):
    if self.version == 0 or not any(self.paletteTypes):
        return b''
    assert len(self.paletteTypes) == len(self.palettes)
    result = bytesjoin([struct.pack('>I', ptype) for ptype in self.paletteTypes])
    assert len(result) == 4 * len(self.palettes)
    return result