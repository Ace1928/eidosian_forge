from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _indent_codestring(self, codestring):
    return '\n'.join([self.tab + line for line in codestring.split('\n')])