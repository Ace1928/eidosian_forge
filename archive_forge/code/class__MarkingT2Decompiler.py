from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
class _MarkingT2Decompiler(psCharStrings.SimpleT2Decompiler):

    def __init__(self, localSubrs, globalSubrs, private):
        psCharStrings.SimpleT2Decompiler.__init__(self, localSubrs, globalSubrs, private)
        for subrs in [localSubrs, globalSubrs]:
            if subrs and (not hasattr(subrs, '_used')):
                subrs._used = set()

    def op_callsubr(self, index):
        self.localSubrs._used.add(self.operandStack[-1] + self.localBias)
        psCharStrings.SimpleT2Decompiler.op_callsubr(self, index)

    def op_callgsubr(self, index):
        self.globalSubrs._used.add(self.operandStack[-1] + self.globalBias)
        psCharStrings.SimpleT2Decompiler.op_callgsubr(self, index)