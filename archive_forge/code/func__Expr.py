import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Expr(self, tree):
    self.fill()
    self.dispatch(tree.value)