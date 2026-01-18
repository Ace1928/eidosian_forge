from collections import namedtuple
from fontTools.cffLib import (
from io import BytesIO
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
from fontTools.ttLib import newTable
from fontTools import varLib
from fontTools.varLib.models import allEqual
from fontTools.misc.roundTools import roundFunc
from fontTools.misc.psCharStrings import T2CharString, T2OutlineExtractor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from functools import partial
from .errors import (
class CFFToCFF2OutlineExtractor(T2OutlineExtractor):
    """This class is used to remove the initial width from the CFF
    charstring without trying to add the width to self.nominalWidthX,
    which is None."""

    def popallWidth(self, evenOdd=0):
        args = self.popall()
        if not self.gotWidth:
            if evenOdd ^ len(args) % 2:
                args = args[1:]
            self.width = self.defaultWidthX
            self.gotWidth = 1
        return args