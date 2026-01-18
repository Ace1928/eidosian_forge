import ast
import copy
import functools
import linecache
import sys
from typing import Any, Dict, List
import triton
class _ForLoopUnroller(ast.NodeTransformer):

    def __init__(self, target, inline_variables, loop_iter):
        self.loop_iter = loop_iter
        self.target = target
        self.inline_variables = inline_variables

    def visit_Name(self, node):
        if node.id != self.target:
            return node
        return ast.Name(str(self.loop_iter))

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Name) and node.slice.id == self.target and isinstance(node.value, ast.Name) and (node.value.id in self.inline_variables):
            return ast.Name(f'{node.value.id}{self.loop_iter}')
        return node