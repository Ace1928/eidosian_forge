from numba.core import errors, ir
from numba.core.rewrites import register_rewrite, Rewrite
@register_rewrite('before-inference')
class RewritePrintCalls(Rewrite):
    """
    Rewrite calls to the print() global function to dedicated IR print() nodes.
    """

    def match(self, func_ir, block, typemap, calltypes):
        self.prints = prints = {}
        self.block = block
        for inst in block.find_insts(ir.Assign):
            if isinstance(inst.value, ir.Expr) and inst.value.op == 'call':
                expr = inst.value
                try:
                    callee = func_ir.infer_constant(expr.func)
                except errors.ConstantInferenceError:
                    continue
                if callee is print:
                    if expr.kws:
                        msg = "Numba's print() function implementation does not support keyword arguments."
                        raise errors.UnsupportedError(msg, inst.loc)
                    prints[inst] = expr
        return len(prints) > 0

    def apply(self):
        """
        Rewrite `var = call <print function>(...)` as a sequence of
        `print(...)` and `var = const(None)`.
        """
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if inst in self.prints:
                expr = self.prints[inst]
                print_node = ir.Print(args=expr.args, vararg=expr.vararg, loc=expr.loc)
                new_block.append(print_node)
                assign_node = ir.Assign(value=ir.Const(None, loc=expr.loc), target=inst.target, loc=inst.loc)
                new_block.append(assign_node)
            else:
                new_block.append(inst)
        return new_block