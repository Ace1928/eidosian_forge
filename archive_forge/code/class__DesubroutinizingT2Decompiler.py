from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
class _DesubroutinizingT2Decompiler(psCharStrings.SimpleT2Decompiler):
    stop_hintcount_ops = ('op_hintmask', 'op_cntrmask', 'op_rmoveto', 'op_hmoveto', 'op_vmoveto')

    def __init__(self, localSubrs, globalSubrs, private=None):
        psCharStrings.SimpleT2Decompiler.__init__(self, localSubrs, globalSubrs, private)

    def execute(self, charString):
        self.need_hintcount = True
        for op_name in self.stop_hintcount_ops:
            setattr(self, op_name, self.stop_hint_count)
        if hasattr(charString, '_desubroutinized'):
            if self.need_hintcount and self.callingStack:
                try:
                    psCharStrings.SimpleT2Decompiler.execute(self, charString)
                except StopHintCountEvent:
                    del self.callingStack[-1]
            return
        charString._patches = []
        psCharStrings.SimpleT2Decompiler.execute(self, charString)
        desubroutinized = charString.program[:]
        for idx, expansion in reversed(charString._patches):
            assert idx >= 2
            assert desubroutinized[idx - 1] in ['callsubr', 'callgsubr'], desubroutinized[idx - 1]
            assert type(desubroutinized[idx - 2]) == int
            if expansion[-1] == 'return':
                expansion = expansion[:-1]
            desubroutinized[idx - 2:idx] = expansion
        if not self.private.in_cff2:
            if 'endchar' in desubroutinized:
                desubroutinized = desubroutinized[:desubroutinized.index('endchar') + 1]
            elif not len(desubroutinized) or desubroutinized[-1] != 'return':
                desubroutinized.append('return')
        charString._desubroutinized = desubroutinized
        del charString._patches

    def op_callsubr(self, index):
        subr = self.localSubrs[self.operandStack[-1] + self.localBias]
        psCharStrings.SimpleT2Decompiler.op_callsubr(self, index)
        self.processSubr(index, subr)

    def op_callgsubr(self, index):
        subr = self.globalSubrs[self.operandStack[-1] + self.globalBias]
        psCharStrings.SimpleT2Decompiler.op_callgsubr(self, index)
        self.processSubr(index, subr)

    def stop_hint_count(self, *args):
        self.need_hintcount = False
        for op_name in self.stop_hintcount_ops:
            setattr(self, op_name, None)
        cs = self.callingStack[-1]
        if hasattr(cs, '_desubroutinized'):
            raise StopHintCountEvent()

    def op_hintmask(self, index):
        psCharStrings.SimpleT2Decompiler.op_hintmask(self, index)
        if self.need_hintcount:
            self.stop_hint_count()

    def processSubr(self, index, subr):
        cs = self.callingStack[-1]
        if not hasattr(cs, '_desubroutinized'):
            cs._patches.append((index, subr._desubroutinized))