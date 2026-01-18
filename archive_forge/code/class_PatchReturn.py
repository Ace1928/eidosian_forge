from pythran.analyses import (ImportedIds, HasReturn, IsAssigned, CFG,
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
class PatchReturn(ast.NodeTransformer):

    def __init__(self, guard, has_break_or_cont):
        self.guard = guard
        self.has_break_or_cont = has_break_or_cont

    def visit_Return(self, node):
        if node is self.guard:
            holder = 'StaticIfNoReturn'
        else:
            holder = 'StaticIfReturn'
        value = node.value
        return ast.Return(ast.Call(ast.Attribute(ast.Attribute(ast.Name('builtins', ast.Load(), None, None), 'pythran', ast.Load()), holder, ast.Load()), [value] if value else [ast.Constant(None, None)], []))