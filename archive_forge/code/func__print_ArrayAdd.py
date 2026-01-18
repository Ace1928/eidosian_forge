from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_ArrayAdd(self, expr):
    return self._expand_fold_binary_op(self._module + '.' + self._add, expr.args)