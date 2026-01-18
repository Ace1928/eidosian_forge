import __future__
import ast
import dis
import inspect
import io
import linecache
import re
import sys
import types
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import islice
from itertools import zip_longest
from operator import attrgetter
from pathlib import Path
from threading import RLock
from tokenize import detect_encoding
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Sized, Tuple, \
class QualnameVisitor(ast.NodeVisitor):

    def __init__(self):
        super(QualnameVisitor, self).__init__()
        self.stack = []
        self.qualnames = {}

    def add_qualname(self, node, name=None):
        name = name or node.name
        self.stack.append(name)
        if getattr(node, 'decorator_list', ()):
            lineno = node.decorator_list[0].lineno
        else:
            lineno = node.lineno
        self.qualnames.setdefault((name, lineno), '.'.join(self.stack))

    def visit_FunctionDef(self, node, name=None):
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)), node
        self.add_qualname(node, name)
        self.stack.append('<locals>')
        children = []
        if isinstance(node, ast.Lambda):
            children = [node.body]
        else:
            children = node.body
        for child in children:
            self.visit(child)
        self.stack.pop()
        self.stack.pop()
        for field, child in ast.iter_fields(node):
            if field == 'body':
                continue
            if isinstance(child, ast.AST):
                self.visit(child)
            elif isinstance(child, list):
                for grandchild in child:
                    if isinstance(grandchild, ast.AST):
                        self.visit(grandchild)
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        assert isinstance(node, ast.Lambda)
        self.visit_FunctionDef(node, '<lambda>')

    def visit_ClassDef(self, node):
        assert isinstance(node, ast.ClassDef)
        self.add_qualname(node)
        self.generic_visit(node)
        self.stack.pop()