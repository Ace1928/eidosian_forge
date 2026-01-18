import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _BinOp(self, t):
    self.write('(')
    self.dispatch(t.left)
    self.write(' ' + self.binop[t.op.__class__.__name__] + ' ')
    self.dispatch(t.right)
    self.write(')')