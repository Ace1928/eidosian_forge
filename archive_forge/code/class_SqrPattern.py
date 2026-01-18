from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class SqrPattern(Pattern):
    pattern = ast.BinOp(left=Placeholder(0), op=ast.Mult(), right=Placeholder(0))

    @staticmethod
    def sub():
        return ast.BinOp(left=Placeholder(0), op=ast.Pow(), right=ast.Constant(2, None))