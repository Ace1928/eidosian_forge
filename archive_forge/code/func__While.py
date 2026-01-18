import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _While(self, t):
    self.fill('while ')
    self.dispatch(t.test)
    self.enter()
    self.dispatch(t.body)
    self.leave()
    if t.orelse:
        self.fill('else')
        self.enter()
        self.dispatch(t.orelse)
        self.leave()