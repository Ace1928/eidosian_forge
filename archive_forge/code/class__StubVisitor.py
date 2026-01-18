import ast
import importlib
import importlib.util
import os
import sys
import threading
import types
import warnings
class _StubVisitor(ast.NodeVisitor):
    """AST visitor to parse a stub file for submodules and submod_attrs."""

    def __init__(self):
        self._submodules = set()
        self._submod_attrs = {}

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level != 1:
            raise ValueError('Only within-module imports are supported (`from .* import`)')
        if node.module:
            attrs: list = self._submod_attrs.setdefault(node.module, [])
            aliases = [alias.name for alias in node.names]
            if '*' in aliases:
                raise ValueError(f'lazy stub loader does not support star import `from {node.module} import *`')
            attrs.extend(aliases)
        else:
            self._submodules.update((alias.name for alias in node.names))