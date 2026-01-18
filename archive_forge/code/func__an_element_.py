from .sage_helper import _within_sage
from .pari import *
import re
def _an_element_(self):
    return Number(RealField(self._precision)(1.0))