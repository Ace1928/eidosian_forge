import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _BoolOp(self, t):
    self.write('(')
    s = ' %s ' % self.boolops[t.op.__class__]
    interleave(lambda: self.write(s), self.dispatch, t.values)
    self.write(')')