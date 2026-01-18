from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class AbsSqrPatternNumpy(AbsSqrPattern):
    pattern = ast.Call(func=ast.Attribute(value=ast.Name(id=mangle('numpy'), ctx=ast.Load(), annotation=None, type_comment=None), attr='square', ctx=ast.Load()), args=[ast.Call(func=ast.Attribute(value=ast.Name(id=mangle('numpy'), ctx=ast.Load(), annotation=None, type_comment=None), attr='abs', ctx=ast.Load()), args=[Placeholder(0)], keywords=[])], keywords=[])