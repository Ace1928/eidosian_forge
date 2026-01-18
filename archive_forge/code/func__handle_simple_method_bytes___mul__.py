from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Symtab import ModuleScope
from Cython.TestUtils import TransformTest
from Cython.Compiler.Visitor import MethodDispatcherTransform
from Cython.Compiler.ParseTreeTransforms import (
def _handle_simple_method_bytes___mul__(self, node, func, args, unbound):
    calls['bytes'] += 1
    return node