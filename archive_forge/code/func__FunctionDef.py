import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _FunctionDef(self, t):
    for deco in t.decorator_list:
        self.fill('@')
        self.dispatch(deco)
    self.fill('def ' + t.name + '(')
    self.dispatch(t.args)
    self.write(')')
    self.enter()
    self.dispatch(t.body)
    self.leave()