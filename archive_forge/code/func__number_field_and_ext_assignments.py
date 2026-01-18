from . import solutionsToPrimeIdealGroebnerBasis
from . import numericalSolutionsToGroebnerBasis
from .component import *
from .coordinates import PtolemyCoordinates
def _number_field_and_ext_assignments(self):
    extensions, assignments = self._extensions_and_assignments()
    if not self._number_field_and_ext_assignments_cache:
        self._number_field_and_ext_assignments_cache = solutionsToPrimeIdealGroebnerBasis.process_extensions_to_pari(extensions)
    return self._number_field_and_ext_assignments_cache