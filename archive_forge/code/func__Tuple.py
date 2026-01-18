import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Tuple(self, t):
    self.write('(')
    if len(t.elts) == 1:
        elt, = t.elts
        self.dispatch(elt)
        self.write(',')
    else:
        interleave(lambda: self.write(', '), self.dispatch, t.elts)
    self.write(')')