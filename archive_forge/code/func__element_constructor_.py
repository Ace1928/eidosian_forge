from .sage_helper import _within_sage
from .pari import *
import re
def _element_constructor_(self, x):
    return Number(RealField(self._precision)(x))