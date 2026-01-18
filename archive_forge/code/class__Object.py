import ast
import sys
import importlib.util
class _Object:
    """Information about Python class or function."""

    def __init__(self, module, name, file, lineno, end_lineno, parent):
        self.module = module
        self.name = name
        self.file = file
        self.lineno = lineno
        self.end_lineno = end_lineno
        self.parent = parent
        self.children = {}
        if parent is not None:
            parent.children[name] = self