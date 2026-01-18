import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Lambda(self, t):
    self.write('(')
    self.write('lambda ')
    self.dispatch(t.args)
    self.write(': ')
    self.dispatch(t.body)
    self.write(')')