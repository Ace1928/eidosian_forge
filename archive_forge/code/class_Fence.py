from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class Fence(Instruction):
    """
    The `fence` instruction.

    As of LLVM 5.0.1:

    fence [syncscope("<target-scope>")] <ordering>  ; yields void
    """
    VALID_FENCE_ORDERINGS = {'acquire', 'release', 'acq_rel', 'seq_cst'}

    def __init__(self, parent, ordering, targetscope=None, name=''):
        super(Fence, self).__init__(parent, types.VoidType(), 'fence', (), name=name)
        if ordering not in self.VALID_FENCE_ORDERINGS:
            msg = 'Invalid fence ordering "{0}"! Should be one of {1}.'
            raise ValueError(msg.format(ordering, ', '.join(self.VALID_FENCE_ORDERINGS)))
        self.ordering = ordering
        self.targetscope = targetscope

    def descr(self, buf):
        if self.targetscope is None:
            syncscope = ''
        else:
            syncscope = 'syncscope("{0}") '.format(self.targetscope)
        fmt = 'fence {syncscope}{ordering}\n'
        buf.append(fmt.format(syncscope=syncscope, ordering=self.ordering))