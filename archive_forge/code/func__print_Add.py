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
def _print_Add(self, expr, order=None):
    terms = self._as_ordered_terms(expr, order=order)
    pforms, indices = ([], [])

    def pretty_negative(pform, index):
        """Prepend a minus sign to a pretty form. """
        if index == 0:
            if pform.height() > 1:
                pform_neg = '- '
            else:
                pform_neg = '-'
        else:
            pform_neg = ' - '
        if pform.binding > prettyForm.NEG or pform.binding == prettyForm.ADD:
            p = stringPict(*pform.parens())
        else:
            p = pform
        p = stringPict.next(pform_neg, p)
        return prettyForm(*p, binding=prettyForm.NEG)
    for i, term in enumerate(terms):
        if term.is_Mul and term.could_extract_minus_sign():
            coeff, other = term.as_coeff_mul(rational=False)
            if coeff == -1:
                negterm = Mul(*other, evaluate=False)
            else:
                negterm = Mul(-coeff, *other, evaluate=False)
            pform = self._print(negterm)
            pforms.append(pretty_negative(pform, i))
        elif term.is_Rational and term.q > 1:
            pforms.append(None)
            indices.append(i)
        elif term.is_Number and term < 0:
            pform = self._print(-term)
            pforms.append(pretty_negative(pform, i))
        elif term.is_Relational:
            pforms.append(prettyForm(*self._print(term).parens()))
        else:
            pforms.append(self._print(term))
    if indices:
        large = True
        for pform in pforms:
            if pform is not None and pform.height() > 1:
                break
        else:
            large = False
        for i in indices:
            term, negative = (terms[i], False)
            if term < 0:
                term, negative = (-term, True)
            if large:
                pform = prettyForm(str(term.p)) / prettyForm(str(term.q))
            else:
                pform = self._print(term)
            if negative:
                pform = pretty_negative(pform, i)
            pforms[i] = pform
    return prettyForm.__add__(*pforms)