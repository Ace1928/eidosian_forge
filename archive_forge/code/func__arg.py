import ast
import io
import sys
import tokenize
def _arg(self, t):
    self.write(t.arg)
    if t.annotation:
        self.write(': ')
        self.dispatch(t.annotation)