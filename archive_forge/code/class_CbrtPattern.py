from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class CbrtPattern(Pattern):
    pattern = ast.BinOp(Placeholder(0), ast.Pow(), ast.Constant(1.0 / 3.0, None))

    @staticmethod
    def sub():
        return ast.Call(func=ast.Attribute(value=ast.Name(id=mangle('numpy'), ctx=ast.Load(), annotation=None, type_comment=None), attr='cbrt', ctx=ast.Load()), args=[Placeholder(0)], keywords=[])
    extra_imports = [ast.Import([ast.alias('numpy', mangle('numpy'))])]