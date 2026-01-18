import pythran.metadata as metadata
import pythran.openmp as openmp
from pythran.utils import isnum
import gast as ast
import os
import sys
import io
def _Constant(self, t):
    if isinstance(t, str):
        return self._Str(t)
    elif t is None or isinstance(t, bool):
        return self._NameConstant(t)
    else:
        return self._Num(t)