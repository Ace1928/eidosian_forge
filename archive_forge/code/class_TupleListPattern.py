from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class TupleListPattern(Pattern):
    pattern = ast.Call(func=ast.Attribute(value=ast.Name('builtins', ast.Load(), None, None), attr='tuple', ctx=ast.Load()), args=[ast.Call(func=ast.Attribute(value=ast.Name('builtins', ast.Load(), None, None), attr='list', ctx=ast.Load()), args=[Placeholder(0)], keywords=[])], keywords=[])

    @staticmethod
    def sub():
        return ast.Call(func=ast.Attribute(value=ast.Name(id='builtins', ctx=ast.Load(), annotation=None, type_comment=None), attr='tuple', ctx=ast.Load()), args=[Placeholder(0)], keywords=[])