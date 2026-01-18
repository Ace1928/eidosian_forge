from llvmlite.ir.transforms import CallVisitor
from numba.core import types
def _rewrite_function(function):
    markpass = _MarkNrtCallVisitor()
    markpass.visit_Function(function)
    for bb in function.basic_blocks:
        for inst in list(bb.instructions):
            if inst in markpass.marked:
                bb.instructions.remove(inst)