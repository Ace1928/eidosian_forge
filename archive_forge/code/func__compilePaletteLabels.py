from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import array
from collections import namedtuple
import struct
import sys
def _compilePaletteLabels(self):
    if self.version == 0 or all((l == self.NO_NAME_ID for l in self.paletteLabels)):
        return b''
    assert len(self.paletteLabels) == len(self.palettes)
    result = bytesjoin([struct.pack('>H', label) for label in self.paletteLabels])
    assert len(result) == 2 * len(self.palettes)
    return result