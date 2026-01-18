from .polynomial import Polynomial
from .component import NonZeroDimensionalComponent
from ..pari import pari
def _get_first(l):
    if l:
        return l[0]
    return None