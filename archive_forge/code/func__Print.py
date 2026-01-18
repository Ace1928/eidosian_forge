import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Print(self, t):
    self.fill('print')
    do_comma = False
    self.write('(')
    for e in t.values:
        if do_comma:
            self.write(', ')
        else:
            do_comma = True
        self.dispatch(e)
    if t.dest:
        self.write(', file=')
        self.dispatch(t.dest)
    if not t.nl:
        self.write(", end=''")
    self.write(')')