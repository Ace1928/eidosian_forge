from .sage_helper import _within_sage
from .pari import *
import re
def _coerce_map_from_(self, S):
    if isinstance(S, RealField_class) or isinstance(S, ComplexField_class):
        prec = min(S.prec(), self._precision)
        return MorphismToSPN(S, self, prec)