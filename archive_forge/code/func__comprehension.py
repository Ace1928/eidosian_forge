import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _comprehension(self, t):
    self.write(' for ')
    self.dispatch(t.target)
    self.write(' in ')
    self.dispatch(t.iter)
    for if_clause in t.ifs:
        self.write(' if ')
        self.dispatch(if_clause)