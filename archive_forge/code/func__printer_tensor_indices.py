import itertools
from sympy.core import S
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Number, Rational
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from sympy.printing.conventions import requires_partial
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.str import sstr
from sympy.utilities.iterables import has_variety
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import hobj, vobj, xobj, \
def _printer_tensor_indices(self, name, indices, index_map={}):
    center = stringPict(name)
    top = stringPict(' ' * center.width())
    bot = stringPict(' ' * center.width())
    last_valence = None
    prev_map = None
    for i, index in enumerate(indices):
        indpic = self._print(index.args[0])
        if (index in index_map or prev_map) and last_valence == index.is_up:
            if index.is_up:
                top = prettyForm(*stringPict.next(top, ','))
            else:
                bot = prettyForm(*stringPict.next(bot, ','))
        if index in index_map:
            indpic = prettyForm(*stringPict.next(indpic, '='))
            indpic = prettyForm(*stringPict.next(indpic, self._print(index_map[index])))
            prev_map = True
        else:
            prev_map = False
        if index.is_up:
            top = stringPict(*top.right(indpic))
            center = stringPict(*center.right(' ' * indpic.width()))
            bot = stringPict(*bot.right(' ' * indpic.width()))
        else:
            bot = stringPict(*bot.right(indpic))
            center = stringPict(*center.right(' ' * indpic.width()))
            top = stringPict(*top.right(' ' * indpic.width()))
        last_valence = index.is_up
    pict = prettyForm(*center.above(top))
    pict = prettyForm(*pict.below(bot))
    return pict