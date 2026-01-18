import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Global(self, t):
    self.fill('global ')
    interleave(lambda: self.write(', '), self.write, t.names)