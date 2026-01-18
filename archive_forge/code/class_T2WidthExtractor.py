from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
class T2WidthExtractor(SimpleT2Decompiler):

    def __init__(self, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private=None, blender=None):
        SimpleT2Decompiler.__init__(self, localSubrs, globalSubrs, private, blender)
        self.nominalWidthX = nominalWidthX
        self.defaultWidthX = defaultWidthX

    def reset(self):
        SimpleT2Decompiler.reset(self)
        self.gotWidth = 0
        self.width = 0

    def popallWidth(self, evenOdd=0):
        args = self.popall()
        if not self.gotWidth:
            if evenOdd ^ len(args) % 2:
                assert self.defaultWidthX is not None, 'CFF2 CharStrings must not have an initial width value'
                self.width = self.nominalWidthX + args[0]
                args = args[1:]
            else:
                self.width = self.defaultWidthX
            self.gotWidth = 1
        return args

    def countHints(self):
        args = self.popallWidth()
        self.hintCount = self.hintCount + len(args) // 2

    def op_rmoveto(self, index):
        self.popallWidth()

    def op_hmoveto(self, index):
        self.popallWidth(1)

    def op_vmoveto(self, index):
        self.popallWidth(1)

    def op_endchar(self, index):
        self.popallWidth()