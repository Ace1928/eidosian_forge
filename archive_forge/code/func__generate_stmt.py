from . import c_ast
def _generate_stmt(self, n, add_indent=False):
    """ Generation from a statement node. This method exists as a wrapper
            for individual visit_* methods to handle different treatment of
            some statements in this context.
        """
    typ = type(n)
    if add_indent:
        self.indent_level += 2
    indent = self._make_indent()
    if add_indent:
        self.indent_level -= 2
    if typ in (c_ast.Decl, c_ast.Assignment, c_ast.Cast, c_ast.UnaryOp, c_ast.BinaryOp, c_ast.TernaryOp, c_ast.FuncCall, c_ast.ArrayRef, c_ast.StructRef, c_ast.Constant, c_ast.ID, c_ast.Typedef, c_ast.ExprList):
        return indent + self.visit(n) + ';\n'
    elif typ in (c_ast.Compound,):
        return self.visit(n)
    elif typ in (c_ast.If,):
        return indent + self.visit(n)
    else:
        return indent + self.visit(n) + '\n'