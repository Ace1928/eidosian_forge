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
def _packTableDirectory(self):
    """Return WOFF2 table directory data."""
    directory = sstruct.pack(self.directoryFormat, self)
    for entry in self.tables.values():
        directory = directory + entry.toString()
    return directory