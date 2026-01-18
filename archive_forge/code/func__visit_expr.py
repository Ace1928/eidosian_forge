from . import c_ast
def _visit_expr(self, n):
    if isinstance(n, c_ast.InitList):
        return '{' + self.visit(n) + '}'
    elif isinstance(n, c_ast.ExprList):
        return '(' + self.visit(n) + ')'
    else:
        return self.visit(n)