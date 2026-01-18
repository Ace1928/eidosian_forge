import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
class RewriteSymbolics(ast.NodeTransformer):

    def visit_Attribute(self, node):
        a = []
        n = node
        while isinstance(n, ast.Attribute):
            a.append(n.attr)
            n = n.value
        if not isinstance(n, ast.Name):
            raise ValueError
        a.append(n.id)
        value = '.'.join(reversed(a))
        return wrap_value(value)

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            raise ValueError()
        return wrap_value(node.id)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if not isinstance(left, ast.Constant) or not isinstance(right, ast.Constant):
            raise ValueError
        if isinstance(node.op, ast.Add):
            return ast.Constant(left.value + right.value)
        elif isinstance(node.op, ast.Sub):
            return ast.Constant(left.value - right.value)
        elif isinstance(node.op, ast.BitOr):
            return ast.Constant(left.value | right.value)
        raise ValueError