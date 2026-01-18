from llvmlite import ir
from llvmlite.ir.transforms import Visitor, CallVisitor
class FastFloatBinOpVisitor(Visitor):
    """
    A pass to add fastmath flag to float-binop instruction if they don't have
    any flags.
    """
    float_binops = frozenset(['fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'fcmp'])

    def __init__(self, flags):
        self.flags = flags

    def visit_Instruction(self, instr):
        if instr.opname in self.float_binops:
            if not instr.flags:
                for flag in self.flags:
                    instr.flags.append(flag)