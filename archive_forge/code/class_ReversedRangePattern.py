from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class ReversedRangePattern(Pattern):
    pattern = ast.Call(func=ast.Attribute(value=ast.Name(id='builtins', ctx=ast.Load(), annotation=None, type_comment=None), attr='reversed', ctx=ast.Load()), args=[ast.Call(func=ast.Attribute(value=ast.Name(id='builtins', ctx=ast.Load(), annotation=None, type_comment=None), attr='range', ctx=ast.Load()), args=[Placeholder(0)], keywords=[])], keywords=[])

    @staticmethod
    def sub():
        return ast.Call(func=ast.Attribute(value=ast.Name(id='builtins', ctx=ast.Load(), annotation=None, type_comment=None), attr='range', ctx=ast.Load()), args=[ast.BinOp(left=Placeholder(0), op=ast.Sub(), right=ast.Constant(1, None)), ast.Constant(-1, None), ast.Constant(-1, None)], keywords=[])