import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Return(self, t):
    self.fill('return')
    if t.value:
        self.write(' ')
        self.dispatch(t.value)