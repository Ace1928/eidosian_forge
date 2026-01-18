import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Assert(self, t):
    self.fill('assert ')
    self.dispatch(t.test)
    if t.msg:
        self.write(', ')
        self.dispatch(t.msg)