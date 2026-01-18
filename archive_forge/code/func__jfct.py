from __future__ import division, print_function, absolute_import
import math
import numpy as np
from ..util import import_
from ..core import RecoverableError
from ..symbolic import ScaledSys
def _jfct(ri, ci):
    if logc:
        return liny[ci] / liny[ri]
    else:
        return 1