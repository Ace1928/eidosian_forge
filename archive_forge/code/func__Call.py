import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Call(self, t):
    self.dispatch(t.func)
    self.write('(')
    comma = False
    for e in t.args:
        if comma:
            self.write(', ')
        else:
            comma = True
        self.dispatch(e)
    for e in t.keywords:
        if comma:
            self.write(', ')
        else:
            comma = True
        self.dispatch(e)
    self.write(')')