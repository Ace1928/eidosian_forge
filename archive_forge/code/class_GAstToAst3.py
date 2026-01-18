from gast.astn import AstToGAst, GAstToAst
import gast
import ast
import sys
class GAstToAst3(GAstToAst):
    if sys.version_info.minor < 10:

        def visit_alias(self, node):
            new_node = ast.alias(self._visit(node.name), self._visit(node.asname))
            return new_node
    if sys.version_info.minor < 9:

        def visit_Subscript(self, node):

            def adjust_slice(s):
                if isinstance(s, ast.Slice):
                    return s
                else:
                    return ast.Index(s)
            if isinstance(node.slice, gast.Tuple):
                if any((isinstance(elt, gast.slice) for elt in node.slice.elts)):
                    new_slice = ast.ExtSlice([adjust_slice(x) for x in self._visit(node.slice.elts)])
                else:
                    value = ast.Tuple(self._visit(node.slice.elts), ast.Load())
                    ast.copy_location(value, node.slice)
                    new_slice = ast.Index(value)
            else:
                new_slice = adjust_slice(self._visit(node.slice))
            ast.copy_location(new_slice, node.slice)
            new_node = ast.Subscript(self._visit(node.value), new_slice, self._visit(node.ctx))
            return ast.copy_location(new_node, node)

    def visit_Assign(self, node):
        new_node = ast.Assign(self._visit(node.targets), self._visit(node.value))
        return ast.copy_location(new_node, node)
    if sys.version_info.minor < 8:

        def visit_Module(self, node):
            new_node = ast.Module(self._visit(node.body))
            return new_node

        def visit_Constant(self, node):
            if node.value is None:
                new_node = ast.NameConstant(node.value)
            elif node.value is Ellipsis:
                new_node = ast.Ellipsis()
            elif isinstance(node.value, bool):
                new_node = ast.NameConstant(node.value)
            elif isinstance(node.value, (int, float, complex)):
                new_node = ast.Num(node.value)
            elif isinstance(node.value, str):
                new_node = ast.Str(node.value)
            else:
                new_node = ast.Bytes(node.value)
            return ast.copy_location(new_node, node)

    def _make_arg(self, node):
        if node is None:
            return None
        if sys.version_info.minor < 8:
            extra_args = tuple()
        else:
            extra_args = (self._visit(node.type_comment),)
        new_node = ast.arg(self._visit(node.id), self._visit(node.annotation), *extra_args)
        return ast.copy_location(new_node, node)

    def visit_Name(self, node):
        new_node = ast.Name(self._visit(node.id), self._visit(node.ctx))
        return ast.copy_location(new_node, node)

    def visit_ExceptHandler(self, node):
        if node.name:
            new_node = ast.ExceptHandler(self._visit(node.type), node.name.id, self._visit(node.body))
            return ast.copy_location(new_node, node)
        else:
            return self.generic_visit(node)
    if sys.version_info.minor < 5:

        def visit_Call(self, node):
            if node.args and isinstance(node.args[-1], gast.Starred):
                args = node.args[:-1]
                starargs = node.args[-1].value
            else:
                args = node.args
                starargs = None
            if node.keywords and node.keywords[-1].arg is None:
                keywords = node.keywords[:-1]
                kwargs = node.keywords[-1].value
            else:
                keywords = node.keywords
                kwargs = None
            new_node = ast.Call(self._visit(node.func), self._visit(args), self._visit(keywords), self._visit(starargs), self._visit(kwargs))
            return ast.copy_location(new_node, node)

        def visit_ClassDef(self, node):
            self.generic_visit(node)
            new_node = ast.ClassDef(name=self._visit(node.name), bases=self._visit(node.bases), keywords=self._visit(node.keywords), body=self._visit(node.body), decorator_list=self._visit(node.decorator_list), starargs=None, kwargs=None)
            return ast.copy_location(new_node, node)
    elif sys.version_info.minor < 8:

        def visit_FunctionDef(self, node):
            new_node = ast.FunctionDef(self._visit(node.name), self._visit(node.args), self._visit(node.body), self._visit(node.decorator_list), self._visit(node.returns))
            return ast.copy_location(new_node, node)

        def visit_AsyncFunctionDef(self, node):
            new_node = ast.AsyncFunctionDef(self._visit(node.name), self._visit(node.args), self._visit(node.body), self._visit(node.decorator_list), self._visit(node.returns))
            return ast.copy_location(new_node, node)

        def visit_For(self, node):
            new_node = ast.For(self._visit(node.target), self._visit(node.iter), self._visit(node.body), self._visit(node.orelse))
            return ast.copy_location(new_node, node)

        def visit_AsyncFor(self, node):
            new_node = ast.AsyncFor(self._visit(node.target), self._visit(node.iter), self._visit(node.body), self._visit(node.orelse), None)
            return ast.copy_location(new_node, node)

        def visit_With(self, node):
            new_node = ast.With(self._visit(node.items), self._visit(node.body))
            return ast.copy_location(new_node, node)

        def visit_AsyncWith(self, node):
            new_node = ast.AsyncWith(self._visit(node.items), self._visit(node.body))
            return ast.copy_location(new_node, node)

        def visit_Call(self, node):
            new_node = ast.Call(self._visit(node.func), self._visit(node.args), self._visit(node.keywords))
            return ast.copy_location(new_node, node)

    def visit_arguments(self, node):
        extra_args = [self._make_arg(node.vararg), [self._make_arg(n) for n in node.kwonlyargs], self._visit(node.kw_defaults), self._make_arg(node.kwarg), self._visit(node.defaults)]
        if sys.version_info.minor >= 8:
            new_node = ast.arguments([self._make_arg(arg) for arg in node.posonlyargs], [self._make_arg(n) for n in node.args], *extra_args)
        else:
            new_node = ast.arguments([self._make_arg(n) for n in node.args], *extra_args)
        return new_node