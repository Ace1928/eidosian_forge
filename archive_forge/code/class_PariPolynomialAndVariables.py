from .polynomial import Polynomial
from .component import NonZeroDimensionalComponent
from ..pari import pari
class PariPolynomialAndVariables:

    def __init__(self, polynomial, variables=None):
        if isinstance(polynomial, Polynomial):
            self.pari_polynomial = pari(polynomial.to_string(lambda x: ('+', '(%s)' % x)))
            self.variables = polynomial.variables()
        else:
            self.pari_polynomial = polynomial
            self.variables = variables

    def get_variable_if_univariate(self):
        if len(self.variables) == 1:
            return self.variables[0]

    def substitute(self, var, value):
        return PariPolynomialAndVariables(polynomial=self.pari_polynomial.substpol(var, value), variables=[v for v in self.variables if not v == var])

    def get_roots(self):
        return self.pari_polynomial.polroots(precision=3.4 * pari.get_real_precision())