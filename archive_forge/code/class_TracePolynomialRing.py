from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
class TracePolynomialRing:
    """
    >>> S = TracePolynomialRing('ab')
    >>> S.var_names
    ['Ta', 'Tb', 'Tab']
    >>> R = TracePolynomialRing('abc')
    >>> R.var_names
    ['Ta', 'Tb', 'Tc', 'Tab', 'Tac', 'Tbc', 'Tabc']
    >>> R('Ta*Tb')
    Tb*Ta
    """

    def __init__(self, gens):
        self._set_var_names(gens)
        self.vars = [pari_poly_variable(v) for v in self.var_names]

    def _set_var_names(self, gens):
        if len(set(gens)) != len(gens) or not set(gens).issubset(string.ascii_lowercase):
            raise ValueError('Generators are unsuitable')
        poly_vars = list(gens) + list(combinations(gens, 2))
        poly_vars += list(combinations(gens, 3))
        self.var_names = ['T' + ''.join(v) for v in poly_vars]

    def __call__(self, poly):
        return pari(poly)