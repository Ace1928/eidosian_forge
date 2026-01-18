from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class StrJoinPattern(Pattern):
    pattern = ast.BinOp(left=ast.BinOp(left=Placeholder(0), op=ast.Add(), right=ast.Constant(Placeholder(1, str), None)), op=ast.Add(), right=Placeholder(2))

    @staticmethod
    def sub():
        return ast.Call(func=ast.Attribute(ast.Attribute(ast.Name('builtins', ast.Load(), None, None), 'str', ast.Load()), 'join', ast.Load()), args=[ast.Constant(Placeholder(1), None), ast.Tuple([Placeholder(0), Placeholder(2)], ast.Load())], keywords=[])