import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def _loose_call(self, value):
    try:
        return self.func(value)
    except ValueError:
        return self.default