from . import solutionsToPrimeIdealGroebnerBasis
from . import numericalSolutionsToGroebnerBasis
from .component import *
from .coordinates import PtolemyCoordinates
def _exact_solution(self):
    extensions, assignments = self._extensions_and_assignments()
    number_field, ext_assignments = self._number_field_and_ext_assignments()
    assignments = solutionsToPrimeIdealGroebnerBasis.update_assignments_and_merge(assignments, ext_assignments)
    return PtolemyCoordinates(assignments, is_numerical=False, py_eval_section=self.py_eval, manifold_thunk=self.manifold_thunk)