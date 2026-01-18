from NumPy.
import os
from pathlib import Path
import ast
import tokenize
import scipy
import pytest
class FindFuncs(ast.NodeVisitor):

    def __init__(self, filename):
        super().__init__()
        self.__filename = filename
        self.bad_filters = []
        self.bad_stacklevels = []

    def visit_Call(self, node):
        p = ParseCall()
        p.visit(node.func)
        ast.NodeVisitor.generic_visit(self, node)
        if p.ls[-1] == 'simplefilter' or p.ls[-1] == 'filterwarnings':
            if node.args[0].value == 'ignore':
                self.bad_filters.append(f'{self.__filename}:{node.lineno}')
        if p.ls[-1] == 'warn' and (len(p.ls) == 1 or p.ls[-2] == 'warnings'):
            if self.__filename == '_lib/tests/test_warnings.py':
                return
            if len(node.args) == 3:
                return
            args = {kw.arg for kw in node.keywords}
            if 'stacklevel' not in args:
                self.bad_stacklevels.append(f'{self.__filename}:{node.lineno}')