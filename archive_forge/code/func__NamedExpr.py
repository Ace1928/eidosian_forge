import ast
import io
import sys
import tokenize
def _NamedExpr(self, tree):
    self.write('(')
    self.dispatch(tree.target)
    self.write(' := ')
    self.dispatch(tree.value)
    self.write(')')