import ast
import io
import sys
import tokenize
def _withitem(self, t):
    self.dispatch(t.context_expr)
    if t.optional_vars:
        self.write(' as ')
        self.dispatch(t.optional_vars)