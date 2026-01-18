from numba.core import errors, ir, types
from numba.core.rewrites import register_rewrite, Rewrite
@register_rewrite('after-inference')
class RewriteStringLiteralGetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `getitem(value=arr, index=$XX)`
    where `$XX` is a StringLiteral value as
    `static_getitem(value=arr, index=<literal value>)`.
    """

    def match(self, func_ir, block, typemap, calltypes):
        """
        Detect all getitem expressions and find which ones have
        string literal indexes
        """
        self.getitems = getitems = {}
        self.block = block
        self.calltypes = calltypes
        for expr in block.find_exprs(op='getitem'):
            if expr.op == 'getitem':
                index_ty = typemap[expr.index.name]
                if isinstance(index_ty, types.StringLiteral):
                    getitems[expr] = (expr.index, index_ty.literal_value)
        return len(getitems) > 0

    def apply(self):
        """
        Rewrite all matching getitems as static_getitems where the index
        is the literal value of the string.
        """
        new_block = ir.Block(self.block.scope, self.block.loc)
        for inst in self.block.body:
            if isinstance(inst, ir.Assign):
                expr = inst.value
                if expr in self.getitems:
                    const, lit_val = self.getitems[expr]
                    new_expr = ir.Expr.static_getitem(value=expr.value, index=lit_val, index_var=expr.index, loc=expr.loc)
                    self.calltypes[new_expr] = self.calltypes[expr]
                    inst = ir.Assign(value=new_expr, target=inst.target, loc=inst.loc)
            new_block.append(inst)
        return new_block