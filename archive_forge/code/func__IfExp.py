import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _IfExp(self, t):
    self.write('(')
    self.dispatch(t.body)
    self.write(' if ')
    self.dispatch(t.test)
    self.write(' else ')
    self.dispatch(t.orelse)
    self.write(')')