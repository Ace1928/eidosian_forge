import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Num(self, t):
    repr_n = repr(t.value)
    if repr_n.startswith('-'):
        self.write('(')
    self.write(repr_n.replace('inf', INFSTR))
    if repr_n.startswith('-'):
        self.write(')')