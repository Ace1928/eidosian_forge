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
def _decodeInstructions(self, glyph):
    glyphStream = self.glyphStream
    instructionStream = self.instructionStream
    instructionLength, glyphStream = unpack255UShort(glyphStream)
    glyph.program = ttProgram.Program()
    glyph.program.fromBytecode(instructionStream[:instructionLength])
    self.glyphStream = glyphStream
    self.instructionStream = instructionStream[instructionLength:]