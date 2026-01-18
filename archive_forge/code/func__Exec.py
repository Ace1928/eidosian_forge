import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Exec(self, t):
    self.fill('exec ')
    self.dispatch(t.body)
    if t.globals:
        self.write(' in ')
        self.dispatch(t.globals)
    if t.locals:
        self.write(', ')
        self.dispatch(t.locals)