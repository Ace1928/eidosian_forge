import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Assign(self, t):
    self.fill()
    for target in t.targets:
        self.dispatch(target)
        self.write(' = ')
    self.dispatch(t.value)