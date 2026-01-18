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
def _get_cs(charstrings, glyphName, filterEmpty=False):
    if glyphName not in charstrings:
        return None
    cs = charstrings[glyphName]
    if filterEmpty:
        cs.decompile()
        if cs.program == []:
            return None
        elif len(cs.program) <= 2 and cs.program[-1] == 'endchar' and (len(cs.program) == 1 or type(cs.program[0]) in (int, float)):
            return None
    return cs