from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _get_letter_generator_for_einsum(self):
    for i in range(97, 123):
        yield chr(i)
    for i in range(65, 91):
        yield chr(i)
    raise ValueError('out of letters')