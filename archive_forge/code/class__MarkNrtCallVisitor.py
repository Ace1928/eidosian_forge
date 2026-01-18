from llvmlite.ir.transforms import CallVisitor
from numba.core import types
class _MarkNrtCallVisitor(CallVisitor):
    """
    A pass to mark all NRT_incref and NRT_decref.
    """

    def __init__(self):
        self.marked = set()

    def visit_Call(self, instr):
        if getattr(instr.callee, 'name', '') in _accepted_nrtfns:
            self.marked.add(instr)