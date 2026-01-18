import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _alias(self, t):
    self.write(t.name)
    if t.asname:
        self.write(' as ' + t.asname)