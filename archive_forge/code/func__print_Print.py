from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_Print(self, prnt):
    print_args = ', '.join((self._print(arg) for arg in prnt.print_args))
    if prnt.format_string != None:
        print_args = '{} % ({})'.format(self._print(prnt.format_string), print_args)
    if prnt.file != None:
        print_args += ', file=%s' % self._print(prnt.file)
    return 'print(%s)' % print_args