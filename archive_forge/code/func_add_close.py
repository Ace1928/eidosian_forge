import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def add_close(x, y, out=None):
    addop = np.add
    it = np.nditer([x, y, out], [], [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    for a, b, c in it:
        addop(a, b, out=c)
    ret = it.operands[2]
    it.close()
    return ret