from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _init_matrix_and_inverse_cache(self):
    if self._matrix_cache and self._inverse_matrix_cache:
        return
    self._matrix_cache, self._inverse_matrix_cache = findLoops.images_of_original_generators(self, penalties=(0, 1, 1))