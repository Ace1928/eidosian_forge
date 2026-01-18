from numba.core import errors, ir
from numba.core.rewrites import register_rewrite, Rewrite
@register_rewrite('before-inference')
class DetectConstPrintArguments(Rewrite):
    """
    Detect and store constant arguments to print() nodes.
    """

    def match(self, func_ir, block, typemap, calltypes):
        self.consts = consts = {}
        self.block = block
        for inst in block.find_insts(ir.Print):
            if inst.consts:
                continue
            for idx, var in enumerate(inst.args):
                try:
                    const = func_ir.infer_constant(var)
                except errors.ConstantInferenceError:
                    continue
                consts.setdefault(inst, {})[idx] = const
        return len(consts) > 0

    def apply(self):
        """
        Store detected constant arguments on their nodes.
        """
        for inst in self.block.body:
            if inst in self.consts:
                inst.consts = self.consts[inst]
        return self.block