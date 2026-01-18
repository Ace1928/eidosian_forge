from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_Declaration(self, decl):
    return '%s = %s' % (self._print(decl.variable.symbol), self._print(decl.variable.value))