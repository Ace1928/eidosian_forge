from . import solutionsToPrimeIdealGroebnerBasis
from . import numericalSolutionsToGroebnerBasis
from .component import *
from .coordinates import PtolemyCoordinates
def _is_zero_dim_prime_and_lex(self):
    is_zero_dim = self.dimension == 0
    is_prime = self.is_prime
    is_lex = self.term_order is None or self.term_order == 'lex'
    return is_zero_dim and is_prime and is_lex