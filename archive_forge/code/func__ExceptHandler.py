import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _ExceptHandler(self, t):
    self.fill('except')
    if t.type:
        self.write(' ')
        self.dispatch(t.type)
    if t.name:
        self.write(' as ')
        self.dispatch(t.name)
    self.enter()
    self.dispatch(t.body)
    self.leave()