import doctest
import re
import textwrap
import numpy as np
def _allclose(self, want, got, rtol=0.001, atol=0.001):
    return np.allclose(want, got, rtol=rtol, atol=atol)