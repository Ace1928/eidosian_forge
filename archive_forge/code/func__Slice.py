import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Slice(self, t):
    if t.lower:
        self.dispatch(t.lower)
    self.write(':')
    if t.upper:
        self.dispatch(t.upper)
    if t.step:
        self.write(':')
        self.dispatch(t.step)