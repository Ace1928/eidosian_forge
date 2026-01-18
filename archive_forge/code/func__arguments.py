import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _arguments(self, t):
    first = True
    defaults = [None] * (len(t.args) - len(t.defaults)) + t.defaults
    for a, d in zip(t.args, defaults):
        if first:
            first = False
        else:
            self.write(', ')
        (self.dispatch(a),)
        if d:
            self.write('=')
            self.dispatch(d)
    if t.vararg:
        if first:
            first = False
        else:
            self.write(', ')
        self.write('*')
        self.write(t.vararg)
    if t.kwarg:
        if first:
            first = False
        else:
            self.write(', ')
        self.write('**' + t.kwarg)