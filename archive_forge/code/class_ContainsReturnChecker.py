import ast
import inspect
import re
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from .. import language
from .._C.libtriton.triton import ir
from ..language import constexpr, tensor
from ..runtime import JITFunction
from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)
class ContainsReturnChecker(ast.NodeVisitor):

    def __init__(self, gscope):
        self.gscope = gscope

    def _visit_stmts(self, body) -> bool:
        for s in body:
            if self.visit(s):
                return True
        return False

    def _visit_function(self, fn) -> bool:
        if isinstance(fn, JITFunction) and (not fn.noinline):
            fn_node = fn.parse()
            return ContainsReturnChecker(self.gscope).visit(fn_node)
        return False

    def generic_visit(self, node) -> bool:
        ret = False
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        ret = ret or self.visit(item)
            elif isinstance(value, ast.AST):
                ret = ret or self.visit(value)
        return ret

    def visit_Attribute(self, node: ast.Attribute) -> bool:
        if isinstance(node.value, ast.Name):
            if node.value.id in self.gscope:
                value = self.gscope[node.value.id]
                fn = getattr(value, node.attr)
                return self._visit_function(fn)
            return False
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> bool:
        if type(node.ctx) == ast.Store:
            return False
        if node.id in self.gscope:
            fn = self.gscope[node.id]
            return self._visit_function(fn)
        return False

    def visit_Return(self, node: ast.Return) -> bool:
        return True

    def visit_Assign(self, node: ast.Assign) -> bool:
        return False

    def visit_AugAssign(self, node: ast.AugAssign) -> bool:
        return False

    def visit_Module(self, node: ast.Module) -> bool:
        return self._visit_stmts(node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> bool:
        return self._visit_stmts(node.body)

    def visit_If(self, node: ast.If) -> bool:
        ret = self._visit_stmts(node.body)
        if node.orelse:
            ret = ret or self._visit_stmts(node.orelse)
        return ret

    def visit_IfExp(self, node: ast.IfExp) -> bool:
        return self.visit(node.body) or self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> bool:
        return self.visit(node.func)