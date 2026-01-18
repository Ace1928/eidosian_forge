from functools import reduce
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError
def _spaces_printer(self, printer, *args):
    spaces_strs = []
    for arg in self.args:
        s = printer._print(arg, *args)
        if isinstance(arg, DirectSumHilbertSpace):
            s = '(%s)' % s
        spaces_strs.append(s)
    return spaces_strs