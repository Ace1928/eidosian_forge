from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Symtab import ModuleScope
from Cython.TestUtils import TransformTest
from Cython.Compiler.Visitor import MethodDispatcherTransform
from Cython.Compiler.ParseTreeTransforms import (
def _handle_simple_method_object___mul__(self, node, func, args, unbound):
    calls['object'] += 1
    return node