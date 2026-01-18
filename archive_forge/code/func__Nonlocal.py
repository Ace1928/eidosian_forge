import ast
import io
import sys
import tokenize
def _Nonlocal(self, t):
    self.fill('nonlocal ')
    interleave(lambda: self.write(', '), self.write, t.names)