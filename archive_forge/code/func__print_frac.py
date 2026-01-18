from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_frac(self, expr):
    return self._print_Mod(Mod(expr.args[0], 1))