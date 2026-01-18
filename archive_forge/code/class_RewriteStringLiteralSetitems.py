from numba.core import errors, ir, types
from numba.core.rewrites import register_rewrite, Rewrite
@register_rewrite('after-inference')
class RewriteStringLiteralSetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `setitem(value=arr, index=$XX, value=)`
    where `$XX` is a StringLiteral value as
    `static_setitem(value=arr, index=<literal value>, value=)`.
    """

    def match(self, func_ir, block, typemap, calltypes):
        """
        Detect all setitem expressions and find which ones have
        string literal indexes
        """
        self.setitems = setitems = {}
        self.block = block
        self.calltypes = calltypes
        for inst in block.find_insts(ir.SetItem):
            index_ty = typemap[inst.index.name]
            if isinstance(index_ty, types.StringLiteral):
                setitems[inst] = (inst.index, index_ty.literal_value)
        return len(setitems) > 0

    def apply(self):
        """
        Rewrite all matching setitems as static_setitems where the index
        is the literal value of the string.
        """
        new_block = ir.Block(self.block.scope, self.block.loc)
        for inst in self.block.body:
            if isinstance(inst, ir.SetItem):
                if inst in self.setitems:
                    const, lit_val = self.setitems[inst]
                    new_inst = ir.StaticSetItem(target=inst.target, index=lit_val, index_var=inst.index, value=inst.value, loc=inst.loc)
                    self.calltypes[new_inst] = self.calltypes[inst]
                    inst = new_inst
            new_block.append(inst)
        return new_block