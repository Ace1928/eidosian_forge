from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_ZeroArray(self, expr):
    return '%s((%s,))' % (self._module_format(self._module + '.' + self._zeros), ','.join(map(self._print, expr.args)))