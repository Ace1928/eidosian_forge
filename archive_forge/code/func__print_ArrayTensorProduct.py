from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _print_ArrayTensorProduct(self, expr):
    letters = self._get_letter_generator_for_einsum()
    contraction_string = ','.join([''.join([next(letters) for j in range(i)]) for i in expr.subranks])
    return '%s("%s", %s)' % (self._module_format(self._module + '.' + self._einsum), contraction_string, ', '.join([self._print(arg) for arg in expr.args]))