import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _AugAssign(self, t):
    self.fill()
    self.dispatch(t.target)
    self.write(' ' + self.binop[t.op.__class__.__name__] + '= ')
    self.dispatch(t.value)