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
class MergeOutlineExtractor(CFFToCFF2OutlineExtractor):
    """Used to extract the charstring commands - including hints - from a
    CFF charstring in order to merge it as another set of region data
    into a CFF2 variable font charstring."""

    def __init__(self, pen, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private=None, blender=None):
        super().__init__(pen, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private, blender)

    def countHints(self):
        args = self.popallWidth()
        self.hintCount = self.hintCount + len(args) // 2
        return args

    def _hint_op(self, type, args):
        self.pen.add_hint(type, args)

    def op_hstem(self, index):
        args = self.countHints()
        self._hint_op('hstem', args)

    def op_vstem(self, index):
        args = self.countHints()
        self._hint_op('vstem', args)

    def op_hstemhm(self, index):
        args = self.countHints()
        self._hint_op('hstemhm', args)

    def op_vstemhm(self, index):
        args = self.countHints()
        self._hint_op('vstemhm', args)

    def _get_hintmask(self, index):
        if not self.hintMaskBytes:
            args = self.countHints()
            if args:
                self._hint_op('vstemhm', args)
            self.hintMaskBytes = (self.hintCount + 7) // 8
        hintMaskBytes, index = self.callingStack[-1].getBytes(index, self.hintMaskBytes)
        return (index, hintMaskBytes)

    def op_hintmask(self, index):
        index, hintMaskBytes = self._get_hintmask(index)
        self.pen.add_hintmask('hintmask', [hintMaskBytes])
        return (hintMaskBytes, index)

    def op_cntrmask(self, index):
        index, hintMaskBytes = self._get_hintmask(index)
        self.pen.add_hintmask('cntrmask', [hintMaskBytes])
        return (hintMaskBytes, index)