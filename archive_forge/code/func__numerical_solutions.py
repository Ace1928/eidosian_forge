from . import solutionsToPrimeIdealGroebnerBasis
from . import numericalSolutionsToGroebnerBasis
from .component import *
from .coordinates import PtolemyCoordinates
def _numerical_solutions(self):
    if not self._is_zero_dim_prime_and_lex():
        raise Exception('Can find solutions only for Groebner basis in lexicographic order of a zero-dimensional ideal.')
    sols = numericalSolutionsToGroebnerBasis.numerical_solutions_with_one(self.polys)

    def process_solution(solution):
        assert isinstance(solution, dict)
        return PtolemyCoordinates(solution, is_numerical=True, py_eval_section=self.py_eval, manifold_thunk=self.manifold_thunk)
    return ZeroDimensionalComponent([process_solution(sol) for sol in sols])