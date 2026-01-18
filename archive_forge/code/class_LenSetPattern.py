from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class LenSetPattern(Pattern):
    pattern = ast.Call(func=ast.Attribute(value=ast.Name('builtins', ast.Load(), None, None), attr='len', ctx=ast.Load()), args=[ast.Call(func=ast.Attribute(value=ast.Name('builtins', ast.Load(), None, None), attr='set', ctx=ast.Load()), args=[Placeholder(0)], keywords=[])], keywords=[])

    @staticmethod
    def sub():
        return ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name('builtins', ast.Load(), None, None), attr='pythran', ctx=ast.Load()), attr='len_set', ctx=ast.Load()), args=[Placeholder(0)], keywords=[])