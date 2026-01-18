from gast.astn import AstToGAst, GAstToAst
import gast
import ast
import sys
class Ast3ToGAst(AstToGAst):
    if sys.version_info.minor < 10:

        def visit_alias(self, node):
            new_node = gast.alias(self._visit(node.name), self._visit(node.asname))
            new_node.lineno = new_node.col_offset = None
            new_node.end_lineno = new_node.end_col_offset = None
            return new_node
    if sys.version_info.minor < 9:

        def visit_ExtSlice(self, node):
            new_node = gast.Tuple(self._visit(node.dims), gast.Load())
            return gast.copy_location(new_node, node)

        def visit_Index(self, node):
            return self._visit(node.value)

        def visit_Assign(self, node):
            new_node = gast.Assign(self._visit(node.targets), self._visit(node.value), None)
            gast.copy_location(new_node, node)
            new_node.end_lineno = new_node.end_col_offset = None
            return new_node
    if sys.version_info.minor < 8:

        def visit_Module(self, node):
            new_node = gast.Module(self._visit(node.body), [])
            return new_node

        def visit_Num(self, node):
            new_node = gast.Constant(node.n, None)
            return gast.copy_location(new_node, node)

        def visit_Ellipsis(self, node):
            new_node = gast.Constant(Ellipsis, None)
            gast.copy_location(new_node, node)
            new_node.end_lineno = new_node.end_col_offset = None
            return new_node

        def visit_Str(self, node):
            new_node = gast.Constant(node.s, None)
            return gast.copy_location(new_node, node)

        def visit_Bytes(self, node):
            new_node = gast.Constant(node.s, None)
            return gast.copy_location(new_node, node)

        def visit_FunctionDef(self, node):
            new_node = gast.FunctionDef(self._visit(node.name), self._visit(node.args), self._visit(node.body), self._visit(node.decorator_list), self._visit(node.returns), None)
            return gast.copy_location(new_node, node)

        def visit_AsyncFunctionDef(self, node):
            new_node = gast.AsyncFunctionDef(self._visit(node.name), self._visit(node.args), self._visit(node.body), self._visit(node.decorator_list), self._visit(node.returns), None)
            return gast.copy_location(new_node, node)

        def visit_For(self, node):
            new_node = gast.For(self._visit(node.target), self._visit(node.iter), self._visit(node.body), self._visit(node.orelse), None)
            return gast.copy_location(new_node, node)

        def visit_AsyncFor(self, node):
            new_node = gast.AsyncFor(self._visit(node.target), self._visit(node.iter), self._visit(node.body), self._visit(node.orelse), None)
            return gast.copy_location(new_node, node)

        def visit_With(self, node):
            new_node = gast.With(self._visit(node.items), self._visit(node.body), None)
            return gast.copy_location(new_node, node)

        def visit_AsyncWith(self, node):
            new_node = gast.AsyncWith(self._visit(node.items), self._visit(node.body), None)
            return gast.copy_location(new_node, node)

        def visit_Call(self, node):
            if sys.version_info.minor < 5:
                if node.starargs:
                    star = gast.Starred(self._visit(node.starargs), gast.Load())
                    gast.copy_location(star, node)
                    starred = [star]
                else:
                    starred = []
                if node.kwargs:
                    kw = gast.keyword(None, self._visit(node.kwargs))
                    gast.copy_location(kw, node.kwargs)
                    kwargs = [kw]
                else:
                    kwargs = []
            else:
                starred = kwargs = []
            new_node = gast.Call(self._visit(node.func), self._visit(node.args) + starred, self._visit(node.keywords) + kwargs)
            return gast.copy_location(new_node, node)

        def visit_NameConstant(self, node):
            if node.value is None:
                new_node = gast.Constant(None, None)
            elif node.value is True:
                new_node = gast.Constant(True, None)
            elif node.value is False:
                new_node = gast.Constant(False, None)
            return gast.copy_location(new_node, node)

        def visit_arguments(self, node):
            new_node = gast.arguments(self._visit(node.args), [], self._visit(node.vararg), self._visit(node.kwonlyargs), self._visit(node.kw_defaults), self._visit(node.kwarg), self._visit(node.defaults))
            return gast.copy_location(new_node, node)

    def visit_Name(self, node):
        new_node = gast.Name(node.id, self._visit(node.ctx), None, None)
        return ast.copy_location(new_node, node)

    def visit_arg(self, node):
        if sys.version_info.minor < 8:
            extra_arg = None
        else:
            extra_arg = self._visit(node.type_comment)
        new_node = gast.Name(node.arg, gast.Param(), self._visit(node.annotation), extra_arg)
        return ast.copy_location(new_node, node)

    def visit_ExceptHandler(self, node):
        if node.name:
            new_node = gast.ExceptHandler(self._visit(node.type), gast.Name(node.name, gast.Store(), None, None), self._visit(node.body))
            return ast.copy_location(new_node, node)
        else:
            return self.generic_visit(node)
    if sys.version_info.minor < 6:

        def visit_comprehension(self, node):
            new_node = gast.comprehension(target=self._visit(node.target), iter=self._visit(node.iter), ifs=self._visit(node.ifs), is_async=0)
            return ast.copy_location(new_node, node)