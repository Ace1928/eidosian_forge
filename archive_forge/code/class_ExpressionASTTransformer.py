from textwrap import dedent
from types import CodeType
import six
from six.moves import builtins
from genshi.core import Markup
from genshi.template.astutil import ASTTransformer, ASTCodeGenerator, parse
from genshi.template.base import TemplateRuntimeError
from genshi.util import flatten
from genshi.compat import ast as _ast, _ast_Constant, get_code_params, \
class ExpressionASTTransformer(TemplateASTTransformer):
    """Concrete AST transformer that implements the AST transformations needed
    for code embedded in templates.
    """

    def visit_Attribute(self, node):
        if not isinstance(node.ctx, _ast.Load):
            return ASTTransformer.visit_Attribute(self, node)
        func = _new(_ast.Name, '_lookup_attr', _ast.Load())
        args = [self.visit(node.value), _new(_ast_Str, node.attr)]
        return _new(_ast.Call, func, args, [])

    def visit_Subscript(self, node):
        if not isinstance(node.ctx, _ast.Load) or not isinstance(node.slice, (_ast.Index, _ast_Constant, _ast.Name, _ast.Call)):
            return ASTTransformer.visit_Subscript(self, node)
        if isinstance(node.slice, (_ast.Name, _ast.Call)):
            slice_value = node.slice
        else:
            slice_value = node.slice.value
        func = _new(_ast.Name, '_lookup_item', _ast.Load())
        args = [self.visit(node.value), _new(_ast.Tuple, (self.visit(slice_value),), _ast.Load())]
        return _new(_ast.Call, func, args, [])