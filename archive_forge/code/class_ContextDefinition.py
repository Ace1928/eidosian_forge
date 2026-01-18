from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class ContextDefinition(Statement):

    def __init__(self, ex_or_in, left=None, right=None, location=None):
        Statement.__init__(self, location)
        self.ex_or_in = ex_or_in
        self.left = left if left is not None else []
        self.right = right if right is not None else []

    def __str__(self):
        res = self.ex_or_in + '\n'
        for coverage in self.left:
            coverage = ''.join((str(c) for c in coverage))
            res += f' LEFT{coverage}\n'
        for coverage in self.right:
            coverage = ''.join((str(c) for c in coverage))
            res += f' RIGHT{coverage}\n'
        res += 'END_CONTEXT'
        return res