from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
class LinesResult(object):

    def __init__(self):
        self.lines = []
        self.s = u''

    def put(self, s):
        self.s += s

    def newline(self):
        self.lines.append(self.s)
        self.s = u''

    def putline(self, s):
        self.put(s)
        self.newline()