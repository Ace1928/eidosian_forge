import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Yield(self, t):
    self.write('(')
    self.write('yield')
    if t.value:
        self.write(' ')
        self.dispatch(t.value)
    self.write(')')