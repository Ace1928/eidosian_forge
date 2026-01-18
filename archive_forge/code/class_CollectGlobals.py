from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
class CollectGlobals(ast.NodeVisitor):

    def __init__(self):
        self.Globals = defaultdict(list)

    def visit_Global(self, node):
        for name in node.names:
            self.Globals[name].append((node, name))