import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Module(self, tree):
    has_init = False
    for stmt in tree.body:
        self.dispatch(stmt)
        if type(stmt) is ast.FunctionDef and stmt.name == '__init__':
            has_init = True
    if has_init:
        self.fill('__init__()')