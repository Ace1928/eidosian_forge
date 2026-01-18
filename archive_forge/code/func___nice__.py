import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
def __nice__(self):
    """common 'paramstr' used by __str__ and __repr__"""
    npstring = np.array2string(self.params, separator=', ')
    paramstr = 'matrix=\n' + textwrap.indent(npstring, '    ')
    return paramstr