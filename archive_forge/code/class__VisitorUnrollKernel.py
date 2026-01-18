import ast
import copy
import functools
import linecache
import sys
from typing import Any, Dict, List
import triton
class _VisitorUnrollKernel(ast.NodeTransformer):

    def __init__(self, N):
        self.inline_variables = set()
        self.N = N

    def visit_AnnAssign(self, node):
        if node.value is None and node.simple == 1 and isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Constant) and (node.annotation.value == 'VAR_ARGS_ARRAY'):
            self.inline_variables.add(node.target.id)
            return []
        if node.value is not None:
            node.value = self.visit(node.value)
        if node.annotation is not None:
            node.annotation = self.visit(node.annotation)
        if node.target is not None:
            node.target = self.visit(node.target)
        return node

    def visit_arguments(self, node):
        new_args = []
        for arg in node.args:
            if arg.annotation is not None and isinstance(arg.annotation, ast.Constant) and (arg.annotation.value == 'VAR_ARGS_ARRAY'):
                self.inline_variables.add(arg.arg)
                new_args += [ast.arg(f'{arg.arg}{i}') for i in range(self.N)]
                continue
            new_args.append(arg)
        if node.vararg is not None:
            self.inline_variables.add(node.vararg.arg)
            new_args += [ast.arg(f'{node.vararg.arg}{i}') for i in range(self.N)]
            node.vararg = None
            new_args += node.kwonlyargs
            node.kwonlyargs = []
        node.args = new_args
        return node

    def visit_For(self, node):
        if not isinstance(node.iter, ast.Call) or node.iter.func.id != 'range' or len(node.iter.args) != 1 or (not isinstance(node.iter.args[0], ast.Call)) or (node.iter.args[0].func.id != 'len') or (len(node.iter.args[0].args) != 1) or (node.iter.args[0].args[0].id not in self.inline_variables):
            node.body = [self.visit(x) for x in node.body]
            return node
        new_nodes = []
        for i in range(self.N):
            unroller = _ForLoopUnroller(target=node.target.id, inline_variables=self.inline_variables, loop_iter=i)
            for body in node.body:
                body = copy.deepcopy(body)
                new_node = ast.fix_missing_locations(unroller.visit(body))
                new_node = self.visit(new_node)
                new_nodes.append(new_node)
        return new_nodes