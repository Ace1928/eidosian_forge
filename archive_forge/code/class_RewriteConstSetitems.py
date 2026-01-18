from numba.core import errors, ir, types
from numba.core.rewrites import register_rewrite, Rewrite
@register_rewrite('before-inference')
class RewriteConstSetitems(Rewrite):
    """
    Rewrite IR statements of the kind `setitem(target=arr, index=$constXX, ...)`
    where `$constXX` is a known constant as
    `static_setitem(target=arr, index=<constant value>, ...)`.
    """

    def match(self, func_ir, block, typemap, calltypes):
        self.setitems = setitems = {}
        self.block = block
        for inst in block.find_insts(ir.SetItem):
            try:
                const = func_ir.infer_constant(inst.index)
            except errors.ConstantInferenceError:
                continue
            setitems[inst] = const
        return len(setitems) > 0

    def apply(self):
        """
        Rewrite all matching setitems as static_setitems.
        """
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if inst in self.setitems:
                const = self.setitems[inst]
                new_inst = ir.StaticSetItem(inst.target, const, inst.index, inst.value, inst.loc)
                new_block.append(new_inst)
            else:
                new_block.append(inst)
        return new_block