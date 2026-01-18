from fontTools.misc.textTools import bytesjoin
from fontTools.misc import sstruct
from . import E_B_D_T_
from .BitmapGlyphMetrics import (
from .E_B_D_T_ import (
import struct
def _removeUnsupportedForColor(dataFunctions):
    dataFunctions = dict(dataFunctions)
    del dataFunctions['row']
    return dataFunctions