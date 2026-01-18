from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
class ASTTransformer(object):
    """General purpose base class for AST transformations.
    
    Every visitor method can be overridden to return an AST node that has been
    altered or replaced in some way.
    """

    def visit(self, node):
        if node is None:
            return None
        if type(node) is tuple:
            return tuple([self.visit(n) for n in node])
        visitor = getattr(self, 'visit_%s' % node.__class__.__name__, None)
        if visitor is None:
            return node
        return visitor(node)

    def _clone(self, node):
        clone = node.__class__()
        for name in getattr(clone, '_attributes', ()):
            try:
                setattr(clone, name, getattr(node, name))
            except AttributeError:
                pass
        for name in clone._fields:
            try:
                value = getattr(node, name)
            except AttributeError:
                pass
            else:
                if value is None:
                    pass
                elif isinstance(value, list):
                    value = [self.visit(x) for x in value]
                elif isinstance(value, tuple):
                    value = tuple((self.visit(x) for x in value))
                else:
                    value = self.visit(value)
                setattr(clone, name, value)
        return clone
    visit_Module = _clone
    visit_Interactive = _clone
    visit_Expression = _clone
    visit_Suite = _clone
    visit_FunctionDef = _clone
    visit_ClassDef = _clone
    visit_Return = _clone
    visit_Delete = _clone
    visit_Assign = _clone
    visit_AugAssign = _clone
    visit_Print = _clone
    visit_For = _clone
    visit_While = _clone
    visit_If = _clone
    visit_With = _clone
    visit_Raise = _clone
    visit_TryExcept = _clone
    visit_TryFinally = _clone
    visit_Try = _clone
    visit_Assert = _clone
    visit_ExceptHandler = _clone
    visit_Import = _clone
    visit_ImportFrom = _clone
    visit_Exec = _clone
    visit_Global = _clone
    visit_Expr = _clone
    visit_BoolOp = _clone
    visit_BinOp = _clone
    visit_UnaryOp = _clone
    visit_Lambda = _clone
    visit_IfExp = _clone
    visit_Dict = _clone
    visit_ListComp = _clone
    visit_GeneratorExp = _clone
    visit_Yield = _clone
    visit_Compare = _clone
    visit_Call = _clone
    visit_Repr = _clone
    visit_Attribute = _clone
    visit_Subscript = _clone
    visit_Name = _clone
    visit_NameConstant = _clone
    visit_List = _clone
    visit_Tuple = _clone
    visit_comprehension = _clone
    visit_excepthandler = _clone
    visit_arguments = _clone
    visit_keyword = _clone
    visit_alias = _clone
    visit_Slice = _clone
    visit_ExtSlice = _clone
    visit_Index = _clone
    del _clone