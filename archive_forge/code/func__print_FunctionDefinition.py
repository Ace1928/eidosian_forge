from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_FunctionDefinition(self, fd):
    body = '\n'.join((self._print(arg) for arg in fd.body))
    return 'def {name}({parameters}):\n{body}'.format(name=self._print(fd.name), parameters=', '.join([self._print(var.symbol) for var in fd.parameters]), body=self._indent_codestring(body))