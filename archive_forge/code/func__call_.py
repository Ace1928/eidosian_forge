from .sage_helper import _within_sage
from .pari import *
import re
def _call_(self, x):
    result = Number(x, precision=self.SPN.precision())
    result._precision = self.target_precision
    return result